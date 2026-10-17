"""会话记忆：把多轮对话存进 PostgreSQL，并在提问时拼成上下文。

为什么不用 Redis 存会话？
- 会话要能被审计（消防问答有时需要追溯"谁在什么时候问过什么"），
  PostgreSQL 天然适合；Redis 只做热缓存与语义缓存。

指代消解很关键：用户问"那它需要几个灭火器？"，
必须先用历史把"它"补全成独立问题再去检索，否则检索必然答偏。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import desc
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import ChatMessage, ChatSession
from app.utils.logging import get_logger

logger = get_logger(__name__)


class SessionStore:
    """会话读写门面。传入 SQLAlchemy Session，便于在事务里使用与测试。"""

    def __init__(self, db: Session):
        self.db = db

    # ---------------- 会话 ----------------
    def get_or_create(self, session_id: str | None, user_id: str = "anonymous") -> ChatSession:
        """按 id 取会话；不存在或未传则新建。"""
        if session_id:
            session = self.db.get(ChatSession, session_id)
            if session:
                return session
            logger.info("session_id=%s 不存在，新建会话", session_id)

        session = ChatSession(user_id=user_id, title="新会话")
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        return session

    def rename_from_first_question(self, session: ChatSession, question: str) -> None:
        """用第一句话给会话命名，方便前端列表展示。"""
        if session.title in {"新会话", ""}:
            session.title = question[:30]
            self.db.commit()

    # ---------------- 消息 ----------------
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        latency_ms: int = 0,
    ) -> ChatMessage:
        """写入一条消息。"""
        msg = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            sources=sources or [],
            latency_ms=latency_ms,
        )
        self.db.add(msg)

        session = self.db.get(ChatSession, session_id)
        if session:
            session.turn_count = (session.turn_count or 0) + 1
        self.db.commit()
        self.db.refresh(msg)
        return msg

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list[dict[str, Any]] | None = None,
        latency_ms: int = 0,
    ) -> None:
        """一次提交写入"一问一答"两条消息。

        为什么要专门有这个批量方法：每条 SQL 都是一次网络往返。
        朴素写法（2× add_message）会产生
        INSERT、UPDATE turn_count、COMMIT × 2 共 6 次往返；
        这里压到 INSERT×2 + UPDATE + COMMIT = 4 次，跨国库上省 1~3 秒。
        """
        self.db.add(ChatMessage(session_id=session_id, role="user", content=question))
        self.db.add(
            ChatMessage(
                session_id=session_id,
                role="assistant",
                content=answer,
                sources=sources or [],
                latency_ms=latency_ms,
            )
        )
        session = self.db.get(ChatSession, session_id)
        if session:
            session.turn_count = (session.turn_count or 0) + 2
        self.db.commit()

    def get_history(self, session_id: str, max_turns: int | None = None) -> list[dict[str, str]]:
        """取最近 N 轮对话，按时间正序返回 [{role, content}]，直接喂给 LLM。

        实现：先按时间倒序取 2*N 条（一问一答），再反转。
        """
        max_turns = max_turns or settings.session_max_turns
        rows = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(desc(ChatMessage.created_at), desc(ChatMessage.id))
            .limit(max_turns * 2)
            .all()
        )
        rows.reverse()
        return [{"role": r.role, "content": r.content} for r in rows if r.role in {"user", "assistant"}]

    def get_history_text(self, session_id: str, max_turns: int | None = None) -> str:
        """历史对话的纯文本形式，用于 condense prompt。"""
        return "\n".join(f"{h['role']}: {h['content']}" for h in self.get_history(session_id, max_turns))

    def clear(self, session_id: str) -> int:
        """清空一个会话的历史消息，返回删除条数。"""
        count = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .delete(synchronize_session=False)
        )
        self.db.commit()
        return count

    # ---------------- 摘要（长会话压缩） ----------------
    def maybe_summarize(self, session: ChatSession, llm: Any = None, threshold: int = 20) -> str:
        """消息数超过阈值时，把早期对话压成摘要存进 sessions.summary。

        TODO(第二批): 摘要在检索前拼进 prompt（"此前讨论过……"），
                      实现真正的长会话记忆而不只是滑窗。
        """
        if not llm or (session.turn_count or 0) < threshold:
            return session.summary or ""
        history = self.get_history_text(session.id, max_turns=threshold)
        try:
            prompt = f"用 200 字以内总结以下消防问答对话的要点：\n{history}"
            response = llm.complete(prompt)
            session.summary = str(getattr(response, "text", response))[:2000]
            self.db.commit()
        except Exception as exc:
            logger.warning("会话摘要失败: %s", exc)
        return session.summary or ""


def count_sessions(db: Session, user_id: str | None = None) -> int:
    """统计会话数（admin 用）。"""
    q = db.query(ChatSession)
    if user_id:
        q = q.filter(ChatSession.user_id == user_id)
    return q.count()
