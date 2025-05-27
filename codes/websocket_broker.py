# websocket_broker.py
import asyncio, json, websockets

SUBSCRIBERS = set()                 # all connected clients


async def broker(ws):               # one coroutine per client
    SUBSCRIBERS.add(ws)
    try:
        async for _ in ws:          # keep the socket open
            pass
    finally:
        SUBSCRIBERS.remove(ws)


async def push(payload: dict):      # send one JSON message to everyone
    if SUBSCRIBERS:
        msg = json.dumps(payload)
        await asyncio.gather(*(ws.send(msg) for ws in SUBSCRIBERS))


def run_ws_server(host="0.0.0.0", port=8765):
    """Return the WebSocket-server coroutine for asyncio.create_task()."""
    return websockets.serve(broker, host, port)
