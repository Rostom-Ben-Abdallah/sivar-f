#!/usr/bin/env python
# detect_live_pose_roi_lips_alerts_only.py
# -------------------------------------------------------------------
# Streams only JSON alerts (no video) over WebSockets to your Flutter AR client.

import cv2
import numpy as np
import math
import mediapipe as mp
import asyncio
import threading
import datetime
import json
import websockets                   # pip install websockets
from ultralytics import YOLO

# ───── WebSocket broker (background thread) ─────────────────────────
SUBS     = set()                   # all connected clients
WS_LOOP  = None                    # will hold the loop

async def _broker(ws):
    SUBS.add(ws)
    try:
        async for _ in ws:          # keep socket alive
            pass
    finally:
        SUBS.remove(ws)

def _start_ws_loop():
    global WS_LOOP
    WS_LOOP = asyncio.new_event_loop()
    asyncio.set_event_loop(WS_LOOP)

    async def _serve_forever():
        await websockets.serve(_broker, '192.168.1.12', 8765)
        print("[WS] Server listening on ws://0.0.0.0:8765")
        await asyncio.Future()      # run forever

    WS_LOOP.run_until_complete(_serve_forever())

threading.Thread(target=_start_ws_loop, daemon=True).start()

async def _broadcast(msg: str):
    if SUBS:
        await asyncio.gather(*(ws.send(msg) for ws in SUBS))

def push_ws(payload: dict):
    """Schedule a JSON message to every open WebSocket."""
    if WS_LOOP and SUBS:
        msg = json.dumps(payload)
        asyncio.run_coroutine_threadsafe(_broadcast(msg), WS_LOOP)

# ───── Vision constants ─────────────────────────────────────────────
CAM_INDICES        = [0, 3]
DISPLAY_WIDTH      = 640
DISPLAY_HEIGHT     = 480

# detection thresholds
CONF_THRESH        = {'cigarette':0.5,'suitcase':0.5,'backpack':0.5,'person':0.6}
LAB_L_MIN          = 200
LAB_AB_DELTA       = 40
WHITE_RATIO        = 0.10
WH_RATIO_THRESH    = 1.2
FALL_ANGLE_THRESH  = 30
LUGGAGE_CONFIRM    = 30

# load models
model_cig   = YOLO(r"C:\Users\ASUS\Downloads\best (10).pt")
model_obj   = YOLO(r"C:\Users\ASUS\Downloads\yolo11n.onnx")
model_pose  = YOLO(r"C:\Users\ASUS\Downloads\yolo11n-pose.onnx")

# map classes
cig_idx         = {v.lower(): k for k,v in model_cig.names.items()}['cigarette']
obj_map         = {v.lower(): k for k,v in model_obj.names.items()}
allowed_names   = ['suitcase','backpack']
allowed_obj_idxs= [obj_map[n] for n in allowed_names]

# face-mesh for lips
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5,
)
LIP_IDX = [61,146,91,181,84,17,314,405,321,375,291]

# drawing
FONT      = cv2.FONT_HERSHEY_SIMPLEX
COLOR_MAP = {'Cigarette':(0,0,255),'Suitcase':(255,0,255),'Backpack':(0,165,255)}

# ───── Helper functions ─────────────────────────────────────────────
def draw_text_stack_outside(img, box, texts_colors,
                            font_scale=0.7, th=2, gap=5):
    H,W = img.shape[:2]
    x1,y1,x2,y2 = box
    sizes = [cv2.getTextSize(t, FONT, font_scale, th)[0] for t,_ in texts_colors]
    w_max = max(w for w,_ in sizes)
    h_sum = sum(h for _,h in sizes) + gap*(len(sizes)-1)
    tx = x2 + 10
    if tx + w_max > W:
        tx = max(0, x1 - w_max - 10)
    can_above = (y1 - 10 - h_sum) >= 0
    baseline = (y1 - 10 - h_sum + sizes[0][1]) if can_above else (y2 + 10 + sizes[0][1])
    for (text,col),(w,h) in zip(texts_colors, sizes):
        cv2.putText(img, text, (tx, int(baseline)), FONT, font_scale, col, th)
        baseline += h + gap

def robust_fall_detection(poses, boxes):
    for (xc,yc,w,h), k in zip(boxes, poses):
        if k.shape[0] < 13 or w/h < WH_RATIO_THRESH:
            continue
        ls = ((k[5][0]+k[6][0])/2, (k[5][1]+k[6][1])/2)
        hb = ((k[11][0]+k[12][0])/2, (k[11][1]+k[12][1])/2)
        angle = abs(math.degrees(math.atan2(hb[1]-ls[1], hb[0]-ls[0])))
        if angle < FALL_ANGLE_THRESH:
            return True, (
                int(xc - w/2), int(yc - h/2),
                int(xc + w/2), int(yc + h/2)
            )
    return False, None

def detect_all(frame):
    dets = []
    # cigarette
    r1 = model_cig.track(
        frame, tracker="botsort.yaml",
        classes=[cig_idx],
        imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT),
        conf=CONF_THRESH['cigarette'], iou=0.4
    )[0]
    for box, conf, tid in zip(
        r1.boxes.xyxy.cpu().numpy(),
        r1.boxes.conf.cpu().numpy(),
        (r1.boxes.id.cpu().numpy()
         if getattr(r1.boxes,'id',None) is not None
         else np.zeros(len(r1),int))
    ):
        if conf >= CONF_THRESH['cigarette']:
            dets.append({'class':'Cigarette',
                         'bbox': box.astype(int).tolist(),
                         'id': int(tid)})
    # suitcase/backpack
    r2 = model_obj.track(
        frame, tracker="botsort.yaml",
        classes=allowed_obj_idxs,
        imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT),
        conf=0.4, iou=0.3
    )[0]
    for box, conf, cls, tid in zip(
        r2.boxes.xyxy.cpu().numpy(),
        r2.boxes.conf.cpu().numpy(),
        r2.boxes.cls.cpu().numpy(),
        (r2.boxes.id.cpu().numpy()
         if getattr(r2.boxes,'id',None) is not None
         else np.zeros(len(r2),int))
    ):
        name = model_obj.names[int(cls)].lower()
        if name in allowed_names and conf >= CONF_THRESH[name]:
            dets.append({'class': name.title(),
                         'bbox': box.astype(int).tolist(),
                         'id': int(tid)})
    return dets

# ───── Main loop ────────────────────────────────────────────────────
def main():
    caps = [(i, cv2.VideoCapture(i)) for i in CAM_INDICES]
    for _, cap in caps:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError(f"Camera {i} open failed")

    # luggage countdown per camera
    luggage = {i: {} for i in CAM_INDICES}

    while True:
        alerts = []
        strip  = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH*len(CAM_INDICES), 3), np.uint8)

        for idx,(cam_idx, cap) in enumerate(caps):
            ok, img = cap.read()
            if not ok:
                continue
            raw = img.copy()
            H,W = raw.shape[:2]

            # YOLO detections
            dets = detect_all(raw)

            # luggage abandonment countdown
            active = set()
            for d in dets:
                if d['class'] in ('Suitcase','Backpack'):
                    tid = d['id']
                    active.add(tid)
                    luggage[cam_idx].setdefault(tid, LUGGAGE_CONFIRM)
                    if luggage[cam_idx][tid] > 0:
                        luggage[cam_idx][tid] -= 1
            luggage[cam_idx] = {k:v for k,v in luggage[cam_idx].items() if k in active}

            # pose + fall detection
            res = model_pose(raw,
                             conf=CONF_THRESH['person'],
                             iou=0.3,
                             imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT))[0]
            boxes_pose = poses = None
            fall_box = None
            if getattr(res,'keypoints',None) and getattr(res.keypoints,'data',None) is not None:
                boxes_pose = res.boxes.xywh.cpu().numpy()
                poses      = res.keypoints.data.cpu().numpy()
                if boxes_pose.size and poses.size:
                    fall, fb = robust_fall_detection(poses, boxes_pose)
                    if fall:
                        fall_box = fb

            # face-mesh lip ROI for smoking
            lips = []
            if boxes_pose is not None:
                for (xc,yc,w,h) in boxes_pose:
                    x1,y1 = int(xc-w/2), int(yc-h/2)
                    x2,y2 = int(xc+w/2), int(yc+h/2)
                    pad = int(0.1*w)
                    fx1,fx2 = max(x1+pad,0), min(x2-pad,W)
                    fy1,fy2 = max(y1,0), min(y1+int(0.6*h),H)
                    roi = raw[fy1:fy2, fx1:fx2]
                    if roi.size == 0:
                        continue
                    up = cv2.resize(roi, ((fx2-fx1)*2, (fy2-fy1)*2))
                    r = mp_face_mesh.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
                    if not r.multi_face_landmarks:
                        continue
                    lm = r.multi_face_landmarks[0]
                    xs = [int(lm.landmark[i].x * up.shape[1]) for i in LIP_IDX]
                    ys = [int(lm.landmark[i].y * up.shape[0]) for i in LIP_IDX]
                    lips.append((
                        (min(xs)//2+fx1, min(ys)//2+fy1,
                         max(xs)//2+fx1, max(ys)//2+fy1),
                        (x1,y1,x2,y2)
                    ))

            # smoking alerts
            smoke_boxes = []
            if lips:
                for d in dets:
                    if d['class'] != 'Cigarette':
                        continue
                    x1,y1,x2,y2 = d['bbox']
                    hit = False
                    for (lx1,ly1,lx2,ly2), pbox in lips:
                        if not (x2<lx1 or x1>lx2 or y2<ly1 or y1>ly2):
                            smoke_boxes.append(pbox)
                            hit = True
                            break
                    if hit:
                        continue
                    # fallback: check if cigarette in torso region – only if poses exist
                    if poses is not None:
                        for i,k in enumerate(poses):
                            if k.shape[0] < 7:
                                continue
                            nx,ny = k[0][:2]
                            neck = (k[5][1] + k[6][1]) / 2
                            mx,my = nx, ny + 0.5*(neck-ny)
                            if x1<=mx<=x2 and y1<=my<=y2:
                                xc,yc,w,h = boxes_pose[i]
                                smoke_boxes.append((
                                    int(xc-w/2), int(yc-h/2),
                                    int(xc+w/2), int(yc+h/2)
                                ))
                                break

            # collect all alerts
            boxes_alert = {}
            if boxes_pose is not None:
                for (xc,yc,w,h) in boxes_pose:
                    x1,y1 = int(xc-w/2),int(yc-h/2)
                    x2,y2 = int(xc+w/2),int(yc+h/2)
                    tor = raw[y1+int(0.3*h):y1+int(0.8*h), x1:x2]
                    if tor.size==0:
                        continue
                    lab = cv2.cvtColor(tor, cv2.COLOR_BGR2LAB)
                    L,A,B = cv2.split(lab)
                    mask = ((L>LAB_L_MIN)&
                            (abs(A-128)<LAB_AB_DELTA)&
                            (abs(B-128)<LAB_AB_DELTA))
                    mask = cv2.morphologyEx(
                        (mask.astype(np.uint8)*255),
                        cv2.MORPH_OPEN, np.ones((3,3),np.uint8)
                    )
                    ratio = cv2.countNonZero(mask)/mask.size
                    lbl = "WITH UNIFORM" if ratio>=WHITE_RATIO else "NO UNIFORM"
                    col = (0,255,0) if lbl=="WITH UNIFORM" else (0,0,255)
                    boxes_alert.setdefault((x1,y1,x2,y2), []).append((lbl,col))

            if fall_box:
                boxes_alert.setdefault(fall_box, []).append(("FALL",(0,0,255)))
            for b in smoke_boxes:
                boxes_alert.setdefault(b, []).append(("SMOKING",(0,0,255)))
            for d in dets:
                if d['class'] in ('suitcase','backpack'):
                    cnt = luggage[cam_idx].get(d['id'],0)
                    bbox=tuple(d['bbox'])
                    if cnt>0:
                        boxes_alert.setdefault(bbox, []).append(
                            (str(cnt), COLOR_MAP[d['class']])
                        )
                    else:
                        boxes_alert.setdefault(bbox, []).append(
                            (f"ABANDONED {d['class'].upper()}", (0,0,255))
                        )

            # draw locally and build JSON alerts
            for box, txts in boxes_alert.items():
                draw_text_stack_outside(raw, box, txts)
                for (label,_col) in txts:
                    alerts.append({"label": label, "bbox": list(box)})

            # composite for local display
            strip[:, idx*DISPLAY_WIDTH:(idx+1)*DISPLAY_WIDTH] = \
                cv2.resize(raw, (DISPLAY_WIDTH, DISPLAY_HEIGHT))
            cv2.putText(strip,
                        f"Camera {idx+1}",
                        (idx*DISPLAY_WIDTH+20, 40),
                        FONT, 1, (255,255,255), 2, cv2.LINE_AA)

        # ── BROADCAST ALERTS ONLY ─────────────────────────────────────
        if alerts:
            push_ws({
                "type":   "alerts",
                "ts":     datetime.datetime.utcnow()
                                            .isoformat(timespec='seconds'),
                "alerts": alerts
            })

        # show local window
        cv2.imshow("Alerts Only (local view)", strip)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    for _ , cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
