# detect_live_pose_roi_lips_dual.py

import cv2
import numpy as np
import math
import mediapipe as mp
from ultralytics import YOLO

# --- USER CONFIGURABLE PARAMETERS ---
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1280, 720

CONF_THRESH = {
    'cigarette': 0.5,
    'knife':      0.2,
    'suitcase':   0.5,
    'backpack':   0.5,
    'person':     0.6
}

# --- Load YOLO models ---
model_cig  = YOLO(r"C:\Users\ASUS\Downloads\best (10).pt")
model_obj  = YOLO(r"C:\Users\ASUS\Downloads\yolo11n.onnx")
model_pose = YOLO(r"C:\Users\ASUS\Downloads\yolo11n-pose.onnx")

# --- Class maps for object tracking ---
cig_map       = {v.lower(): k for k, v in model_cig.names.items()}
cig_idx       = cig_map['cigarette']
obj_map       = {v.lower(): k for k, v in model_obj.names.items()}
allowed_names = ['knife', 'suitcase', 'backpack']
allowed_obj_idxs = [obj_map[n] for n in allowed_names]

# --- MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
LIP_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]

# --- Visualization colors ---
COLOR_MAP = {
    'Cigarette': (0,0,255),
    'Knife':     (0,255,255),
    'Suitcase':  (255,0,255),
    'Backpack':  (0,165,255),
    'Fall':      (0,0,0),
    'Smoking':   (0,128,0)
}


def robust_fall_detection(poses, boxes):
    for (xc, yc, w, h), kpts in zip(boxes, poses):
        head_y     = kpts[0][1]
        ls_x, ls_y = kpts[5][:2]
        rs_x, rs_y = kpts[6][:2]
        lb_x, lb_y = kpts[11][:2]
        rb_x, rb_y = kpts[12][:2]
        lf_y       = kpts[15][1]
        rf_y       = kpts[16][1]
        if w / h < 1.2:
            continue
        len_torso = math.hypot(ls_x - lb_x, ls_y - lb_y)
        ankle_y   = min(lf_y, rf_y)
        if head_y < ankle_y - 0.3 * len_torso:
            continue
        sh_x = (ls_x + rs_x) / 2
        sh_y = (ls_y + rs_y) / 2
        hp_x = (lb_x + rb_x) / 2
        hp_y = (lb_y + rb_y) / 2
        angle = abs(math.degrees(math.atan2(hp_y - sh_y, sh_x - hp_x)))
        if angle > 45:
            continue
        xmin = int(xc - w/2)
        ymin = int(yc - h/2)
        xmax = int(xc + w/2)
        ymax = int(yc + h/2)
        return True, (xmin, ymin, xmax, ymax)
    return False, None


def detect_all(frame):
    dets = []

    # Cigarette tracking
    r1 = model_cig.track(
        frame,
        tracker="botsort.yaml",
        classes=[cig_idx],
        imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT),
        conf=CONF_THRESH['cigarette'],
        iou=0.3,
        augment=True
    )[0]
    xy1 = r1.boxes.xyxy.cpu().numpy()
    cf1 = r1.boxes.conf.cpu().numpy()
    id1 = (r1.boxes.id.cpu().numpy() 
           if getattr(r1.boxes, 'id', None) is not None 
           else np.zeros((len(xy1),), dtype=int))
    for box, conf, tid in zip(xy1, cf1, id1):
        if conf < CONF_THRESH['cigarette']:
            continue
        x1, y1, x2, y2 = map(int, box)
        dets.append({
            'class': 'Cigarette',
            'score': float(conf),
            'bbox':  [x1, y1, x2, y2],
            'id':    int(tid)
        })

    # Knife / Suitcase / Backpack tracking
    r2 = model_obj.track(
        frame,
        tracker="botsort.yaml",
        classes=allowed_obj_idxs,
        imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT),
        conf=0.1,
        iou=0.3,
        augment=True
    )[0]
    xy2  = r2.boxes.xyxy.cpu().numpy()
    cf2  = r2.boxes.conf.cpu().numpy()
    cls2 = r2.boxes.cls.cpu().numpy()
    id2  = (r2.boxes.id.cpu().numpy() 
            if getattr(r2.boxes, 'id', None) is not None 
            else np.zeros((len(xy2),), dtype=int))
    for box, conf, cls, tid in zip(xy2, cf2, cls2, id2):
        name = model_obj.names[int(cls)].lower()
        if name not in allowed_names or conf < CONF_THRESH[name]:
            continue
        x1, y1, x2, y2 = map(int, box)
        dets.append({
            'class': name.title(),
            'score': float(conf),
            'bbox':  [x1, y1, x2, y2],
            'id':    int(tid)
        })

    return dets


def annotate_frame(frame):
    H, W = frame.shape[:2]
    raw = frame.copy()

    fall_ctr = knife_ctr = obj_ctr = smoke_ctr = 0
    obj_type = None

    # 1) detections
    dets = detect_all(raw)
    if any(d['class'] == 'Knife' for d in dets):
        knife_ctr = 30
    carried = [d['class'] for d in dets if d['class'] in ('Suitcase', 'Backpack')]
    if carried:
        obj_type, obj_ctr = carried[0], 30

    # 2) pose
    res_pose = model_pose(raw,
                          conf=CONF_THRESH['person'],
                          iou=0.3,
                          imgsz=max(DISPLAY_WIDTH, DISPLAY_HEIGHT)
                          )[0]
    frame = res_pose.plot()
    boxes_pose = poses = None
    if res_pose.keypoints is not None and hasattr(res_pose.keypoints, 'data'):
        boxes_pose = res_pose.boxes.xywh.cpu().numpy()
        poses      = res_pose.keypoints.data.cpu().numpy()
        if poses.shape[0] > 0:
            fell, _ = robust_fall_detection(poses, boxes_pose)
            if fell:
                fall_ctr = 30

    # 3) lip mesh
    lip_boxes = []
    if boxes_pose is not None:
        for (xc, yc, w_bb, h_bb) in boxes_pose:
            x1_bb = int(xc - w_bb/2)
            y1_bb = int(yc - h_bb/2)
            x2_bb = int(xc + w_bb/2)
            y2_bb = int(yc + h_bb/2)
            pad_w = int(0.1 * w_bb)
            fx1 = max(x1_bb + pad_w, 0)
            fx2 = min(x2_bb - pad_w, W)
            fy1 = max(y1_bb, 0)
            fy2 = min(y1_bb + int(0.6 * h_bb), H)
            roi = frame[fy1:fy2, fx1:fx2]
            if roi.size == 0:
                continue
            up = cv2.resize(
                roi,
                ((fx2 - fx1) * 2, (fy2 - fy1) * 2),
                interpolation=cv2.INTER_LINEAR
            )
            mp_res = mp_face_mesh.process(cv2.cvtColor(up, cv2.COLOR_BGR2RGB))
            if mp_res.multi_face_landmarks:
                lm = mp_res.multi_face_landmarks[0]
                xs = [int(lm.landmark[i].x * up.shape[1]) for i in LIP_IDX]
                ys = [int(lm.landmark[i].y * up.shape[0]) for i in LIP_IDX]
                gx = [fx1 + x // 2 for x in xs]
                gy = [fy1 + y // 2 for y in ys]
                lx1, ly1 = min(gx), min(gy)
                lx2, ly2 = max(gx), max(gy)
                lip_boxes.append((lx1, ly1, lx2, ly2))
                for (mx, my) in zip(gx, gy):
                    cv2.circle(frame, (mx, my), 2, (0, 255, 0), -1)
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)

    # 4) smoking alert
    for det in dets:
        if det['class'] != 'Cigarette':
            continue
        x1, y1, x2, y2 = det['bbox']
        hit = False
        for (lx1, ly1, lx2, ly2) in lip_boxes:
            if not (x2 < lx1 or x1 > lx2 or y2 < ly1 or y1 > ly2):
                hit = True
                break
        if not hit and poses is not None:
            for kpts in poses:
                if kpts.shape[0] < 7:
                    continue
                nose_x, nose_y = kpts[0][:2]
                ls_y, rs_y    = kpts[5][1], kpts[6][1]
                neck_y        = (ls_y + rs_y) / 2
                mouth_x       = nose_x
                mouth_y       = nose_y + 0.5 * (neck_y - nose_y)
                if x1 <= mouth_x <= x2 and y1 <= mouth_y <= y2:
                    hit = True
                    break
        if hit:
            smoke_ctr = 30
            break

    # 5) draw detections
    for d in dets:
        cls, sc, bb, tid = d['class'], d['score'], d['bbox'], d['id']
        x1, y1, x2, y2 = bb
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      COLOR_MAP.get(cls, (0,255,0)), 2)
        cv2.putText(frame, f"{cls} {sc:.2f} ID:{tid}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, COLOR_MAP.get(cls, (0,255,0)), 2)

    # banners
    banners = [
        (fall_ctr,  "🚨 FALL DETECTED 🚨",  0,   (0,0,255)),
        (smoke_ctr, "🚬 SMOKING ALERT 🚬",  60,  (0,128,0)),
        (knife_ctr, "⚠️ KNIFE DETECTED ⚠️",120, (0,0,255)),
    ]
    for cnt, text, y0, col in banners:
        if cnt > 0:
            ov = frame.copy()
            cv2.rectangle(ov, (0,y0), (W,y0+60), (255,255,255), -1)
            cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, text, (W//2 - 200, y0 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, col, 3)

    if obj_ctr > 0 and obj_type:
        ov = frame.copy()
        cv2.rectangle(ov, (0,180), (W,240), (255,255,255), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        msg = f"🧳 ABANDONED {obj_type.upper()} 🧳"
        cv2.putText(frame, msg, (W//2 - 220, 225),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    return frame


def main():
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(3)
    for cap in (cap1, cap2):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        if not cap.isOpened():
            raise RuntimeError("Cannot open one of the cameras")

    while True:
        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not (ret1 and ret2):
            break

        a1 = annotate_frame(f1)
        a2 = annotate_frame(f2)

        combined = np.hstack((a1, a2))
        combined = cv2.resize(combined, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        cv2.imshow("Camera 0 | Camera 1", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
