"""
会话注册表（轻量 SQLite，标准库实现）
维护会话列表所需的索引元数据：创建时间、消息数、最近用户问题、最近活跃时间。
对话内容本体由 LangGraph checkpointer 持久化，这里不存消息正文，二者职责分离。
"""

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

from config import CHECKPOINT_DB_PATH

_DB_DIR = os.path.dirname(CHECKPOINT_DB_PATH) or "."
REGISTRY_DB_PATH = os.path.join(_DB_DIR, "sessions.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    created_at        REAL NOT NULL,
    last_activity     REAL NOT NULL,
    message_count     INTEGER NOT NULL DEFAULT 0,
    last_user_question TEXT NOT NULL DEFAULT ''
);
"""

_lock = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        os.makedirs(_DB_DIR, exist_ok=True)
        _conn = sqlite3.connect(REGISTRY_DB_PATH, check_same_thread=False)
        _conn.execute(_SCHEMA)
        _conn.commit()
    return _conn


def ensure_session(session_id: str) -> None:
    """会话不存在则创建（首次发言时登记）"""
    now = time.time()
    with _lock:
        conn = _get_conn()
        conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, last_activity) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        conn.commit()


def record_turn(session_id: str, user_message: str) -> None:
    """一轮对话结束后更新索引：消息数 +2（用户+助手），登记最近用户问题"""
    ensure_session(session_id)
    with _lock:
        conn = _get_conn()
        conn.execute(
            """UPDATE sessions
               SET message_count = message_count + 2,
                   last_user_question = ?,
                   last_activity = ?
               WHERE session_id = ?""",
            (user_message, time.time(), session_id),
        )
        conn.commit()


def list_sessions() -> List[Dict[str, Any]]:
    with _lock:
        rows = _get_conn().execute(
            """SELECT session_id, created_at, message_count, last_user_question
               FROM sessions ORDER BY last_activity DESC"""
        ).fetchall()
    return [
        {
            "session_id": r[0],
            "created_at": r[1],
            "message_count": r[2],
            "last_user_question": r[3],
        }
        for r in rows
    ]


def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        row = _get_conn().execute(
            """SELECT session_id, created_at, message_count, last_user_question
               FROM sessions WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row[0],
        "created_at": row[1],
        "message_count": row[2],
        "last_user_question": row[3],
    }


def delete_session(session_id: str) -> bool:
    with _lock:
        cur = _get_conn().execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        _get_conn().commit()
    return cur.rowcount > 0
