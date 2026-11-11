import json
import queue
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import Request


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"
    CLOSE = "__close__"


_session_stream: Dict[str, "queue.Queue"] = {}

def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)

def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列"""
    q = queue.Queue()
    _session_stream[session_id] = q
    return q

def remove_sse_queue(session_id: str):
    """移除指定 session 的队列"""
    _session_stream.pop(session_id, None)

def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"

def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """
    通过 session_id 推送事件
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})

async def sse_generator(session_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    SSE 生成器，用于 FastAPI 的 StreamingResponse
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack("ready", {})

        while True:
            if await request.is_disconnected():
                break

            try:
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            if event == "__close__":
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        return
    finally:
        remove_sse_queue(session_id)
