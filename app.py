#!/usr/bin/env python3
"""
智能客服工作台 - FastAPI 单服务入口
LangGraph 图进程内运行，SQLite checkpointer 持久化对话状态，
SSE 流式聊天；对外接口契约与前端 templates/index.html 对齐。
"""

import json
import os
import sqlite3
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel

from config import (
    APP_HOST, APP_PORT, CHECKPOINT_DB_PATH, OPENAI_API_KEY, OPENAI_MODEL,
    SYSTEM_NAME, VERSION,
)
import session_store
from multi_agent_customer_service import make_graph

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 全局单例：检查点连接 + 编译后的图
_checkpoint_conn: Optional[sqlite3.Connection] = None
checkpointer: Optional[SqliteSaver] = None
graph = None


def get_graph():
    """惰性构建图（带 checkpointer），进程内单例"""
    global _checkpoint_conn, checkpointer, graph
    if graph is None:
        os.makedirs(os.path.dirname(CHECKPOINT_DB_PATH) or ".", exist_ok=True)
        _checkpoint_conn = sqlite3.connect(
            CHECKPOINT_DB_PATH, check_same_thread=False
        )
        checkpointer = SqliteSaver(_checkpoint_conn)
        graph = make_graph(checkpointer=checkpointer)
    return graph


@asynccontextmanager
async def lifespan(_: FastAPI):
    # 启动时预热：建库建表、编译图，暴露配置错误而不是首请求才炸
    get_graph()
    session_store.ensure_session("__init__")
    session_store.delete_session("__init__")
    yield


app = FastAPI(title=SYSTEM_NAME, version=VERSION, lifespan=lifespan)


# ---------------------------------------------------------------- health

@app.get("/")
def index():
    return FileResponse(os.path.join(_BASE_DIR, "templates", "index.html"))


@app.get("/api/health")
def health():
    return {"status": "healthy", "version": VERSION}


@app.get("/api/test")
def test():
    """环境与依赖自检：图编译、LLM 配置、checkpointer 可写"""
    g = get_graph()
    return {
        "status": "ok",
        "graph_compiled": g is not None,
        "llm_configured": bool(OPENAI_API_KEY),
        "model": OPENAI_MODEL,
        "checkpointer": "sqlite",
        "version": VERSION,
    }


# ---------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


def _run_graph(message: str, session_id: str) -> Dict[str, Any]:
    """以 session_id 为 thread_id 跑一轮图，返回最终状态"""
    g = get_graph()
    config = {"configurable": {"thread_id": session_id}}
    final: Dict[str, Any] = g.invoke(
        {"customer_query": message, "session_id": session_id}, config
    )
    session_store.record_turn(session_id, message)
    return final


@app.post("/api/chat")
def chat(req: ChatRequest):
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is empty")

    session_id = req.session_id or "default"
    try:
        final = _run_graph(message, session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"graph execution failed: {e}")

    return {
        "response": final.get("response", ""),
        "agent": final.get("current_agent", "智能客服"),
        "query_type": final.get("query_type", ""),
        "session_id": session_id,
        "thread_id": session_id,
    }


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """SSE 流式聊天：增量推送专家回复 token，最后推送元信息与结束事件"""
    message = (req.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="message is empty")

    session_id = req.session_id or "default"

    def sse(payload: Dict[str, Any]) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def event_gen():
        g = get_graph()
        config = {"configurable": {"thread_id": session_id}}
        try:
            # stream_mode="messages" 产出 (AIMessageChunk, metadata)；
            # 过滤掉意图分类节点的内部调用，只向客户端透出业务回答的增量
            for chunk, meta in g.stream(
                {"customer_query": message, "session_id": session_id},
                config,
                stream_mode="messages",
            ):
                if meta.get("langgraph_node") == "classify_query":
                    continue
                content = getattr(chunk, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        p.get("text", "") for p in content if isinstance(p, dict)
                    )
                if content:
                    yield sse({"type": "delta", "content": content})

            snapshot = g.get_state(config)
            values = snapshot.values if snapshot else {}
            yield sse({
                "type": "final",
                "response": values.get("response", ""),
                "agent": values.get("current_agent", "智能客服"),
                "query_type": values.get("query_type", ""),
                "session_id": session_id,
            })
            session_store.record_turn(session_id, message)
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
        yield sse({"type": "done"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------- sessions

@app.get("/api/sessions")
def get_sessions():
    return {"sessions": session_store.list_sessions()}


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    info = session_store.get_session_info(session_id)
    if info is None:
        raise HTTPException(status_code=404, detail="session not found")

    g = get_graph()
    snapshot = g.get_state({"configurable": {"thread_id": session_id}})
    values = snapshot.values if snapshot else {}
    history = values.get("persisted_dialogue") or []

    return {
        "session": {
            **info,
            "conversation_history": history,
        }
    }


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    deleted = session_store.delete_session(session_id)
    try:
        checkpointer.delete_thread(session_id)  # 清除 checkpointer 中的对话状态
    except Exception as e:
        # 注册表已删即可；checkpointer 残留不影响功能
        print(f"[delete_session] checkpoint cleanup skipped: {e}")
    return {"message": "会话删除成功", "deleted": deleted}


def main():
    import uvicorn

    print(f"{SYSTEM_NAME} v{VERSION}")
    print(f"http://{APP_HOST}:{APP_PORT}")
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()
