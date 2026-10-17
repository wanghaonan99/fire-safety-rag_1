'use client';

/** 会话列表 Tab：浏览历史会话、继续对话、删除。 */

import { useCallback, useEffect, useState } from 'react';

import { api, formatTime } from '../lib/api';

export default function SessionsTab({ user, activeId, onOpen, toast, refreshKey }) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busyId, setBusyId] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const rows = await api(`/api/sessions?user_id=${encodeURIComponent(user)}&limit=50`);
      setSessions(rows || []);
    } catch (err) {
      toast?.(`读取会话失败：${err.message}`, 'err');
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, [user, toast]);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  async function remove(session) {
    if (!window.confirm(`删除会话「${session.title}」及其所有消息？此操作不可撤销。`)) return;
    setBusyId(session.id);
    try {
      await api(`/api/sessions/${session.id}`, { method: 'DELETE' });
      toast?.('会话已删除', 'ok');
      setSessions((s) => s.filter((x) => x.id !== session.id));
      if (session.id === activeId) onOpen(null);
    } catch (err) {
      toast?.(`删除失败：${err.message}`, 'err');
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="shell">
      <div className="card">
        <div className="card-head">
          <h3>会话记录</h3>
          <span className="desc">共 {sessions.length} 个 · 按最近更新排序</span>
          <div className="spacer" />
          <button className="btn sm" onClick={load} disabled={loading}>
            {loading ? '刷新中…' : '↻ 刷新'}
          </button>
        </div>

        {loading && sessions.length === 0 ? (
          <div className="empty">正在加载…</div>
        ) : sessions.length === 0 ? (
          <div className="empty">
            <div className="big">💬</div>
            还没有会话记录<br />
            <span style={{ fontSize: 13 }}>去「问答」页问第一个问题吧</span>
          </div>
        ) : (
          <div className="list">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`item clickable${s.id === activeId ? ' active' : ''}`}
                onClick={() => onOpen(s.id)}
              >
                <div className="main">
                  <div className="name">{s.title || '新会话'}</div>
                  <div className="sub">
                    <span>{formatTime(s.updated_at)}</span>
                    <span>· {s.message_count} 条消息</span>
                    {s.preview && <span style={{ color: 'var(--ink-3)' }}>· {s.preview}…</span>}
                  </div>
                </div>
                <div className="actions" onClick={(e) => e.stopPropagation()}>
                  <button className="btn sm" onClick={() => onOpen(s.id)}>继续</button>
                  <button
                    className="btn sm danger"
                    disabled={busyId === s.id}
                    onClick={() => remove(s)}
                  >
                    {busyId === s.id ? '删除中' : '删除'}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
