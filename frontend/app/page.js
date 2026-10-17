'use client';

/**
 * 应用外壳：顶栏 + 主视图（问答 / 资料库）+ 两个抽屉（会话 / 设置）。
 *
 * 信息架构（P1 改造）：
 *   原来是「问答 / 会话 / 资料库 / 设置」四个平级 tab；
 *   现在改成「1 个主任务 + 3 个支撑入口」——
 *     · 主视图：问答（默认） / 资料库（二级页，从顶栏"资料徽标"进入）
 *     · 会话：左侧抽屉
 *     · 设置：右侧抽屉
 *   顶栏同时曝光"资料覆盖"（N 份 · M 块），让用户知道库的边界，减少超范围提问。
 *
 * 本地状态（localStorage，不上传）：user_id、检索参数、管理口令、主题、字号。
 */

import { useCallback, useEffect, useState } from 'react';

import ChatTab from './components/ChatTab';
import DocsTab from './components/DocsTab';
import Drawer from './components/Drawer';
import SessionsTab from './components/SessionsTab';
import SettingsTab from './components/SettingsTab';
import { api, getUserId, loadLocal, saveLocal } from './lib/api';

const DEFAULT_SETTINGS = {
  topK: 5,
  useCache: true,
  useQueryTransform: false,
  docCategory: '',
};

const FONT_STEPS = [
  { v: 0.94, label: '小' },
  { v: 1, label: '中' },
  { v: 1.08, label: '大' },
];

const THEME_STEPS = [
  { v: 'auto', label: '跟随系统', icon: '◐' },
  { v: 'light', label: '浅色', icon: '☀' },
  { v: 'dark', label: '深色', icon: '☾' },
];

function resolveTheme(theme) {
  if (theme !== 'auto') return theme;
  if (typeof window === 'undefined') return 'light';
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function Home() {
  const [view, setView] = useState('chat'); // chat | docs
  const [drawer, setDrawer] = useState(null); // null | sessions | settings
  const [health, setHealth] = useState(null);
  const [user, setUser] = useState('anonymous');
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [adminToken, setAdminToken] = useState('');
  const [sessionId, setSessionId] = useState(null);
  const [sessionsKey, setSessionsKey] = useState(0);
  const [sessionCount, setSessionCount] = useState(0);
  const [docsStat, setDocsStat] = useState({ docs: 0, chunks: 0, loading: true });
  const [theme, setTheme] = useState('auto');
  const [fontScale, setFontScale] = useState(1);
  const [bgOn, setBgOn] = useState(true);
  const [toasts, setToasts] = useState([]);

  const toast = useCallback((message, kind = 'info') => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, message, kind }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 5200);
  }, []);

  // ---------- 启动：读本地偏好 ----------
  useEffect(() => {
    setUser(getUserId());
    setSettings({ ...DEFAULT_SETTINGS, ...loadLocal('fsrag.settings', {}) });
    setAdminToken(loadLocal('fsrag.admin_token', '') || '');
    setTheme(loadLocal('fsrag.theme', 'auto'));
    setBgOn(loadLocal('fsrag.bg', 'on') !== 'off');
    const fs = Number(loadLocal('fsrag.font_scale', 1));
    setFontScale(FONT_STEPS.some((s) => s.v === fs) ? fs : 1);
  }, []);

  // 背景图开关（弱光/投影环境可关掉，纯渐变底）
  useEffect(() => {
    document.documentElement.dataset.bg = bgOn ? 'on' : 'off';
    saveLocal('fsrag.bg', bgOn ? 'on' : 'off');
  }, [bgOn]);

  // ---------- 主题与字号 ----------
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = resolveTheme(theme);
    root.style.setProperty('--fs-boost', String(fontScale));
    saveLocal('fsrag.theme', theme);
    saveLocal('fsrag.font_scale', fontScale);

    // 跟随系统时，系统切换要跟着变
    if (theme !== 'auto') return undefined;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => {
      root.dataset.theme = mq.matches ? 'dark' : 'light';
    };
    mq.addEventListener('change', onChange);
    return () => mq.removeEventListener('change', onChange);
  }, [theme, fontScale]);

  // ---------- 健康检查 ----------
  const checkHealth = useCallback(async () => {
    try {
      const h = await api('/health', { timeout: 8000 });
      setHealth(h);
      return h;
    } catch {
      setHealth(null);
      return null;
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const t = setInterval(checkHealth, 30000);
    return () => clearInterval(t);
  }, [checkHealth]);

  // ---------- 会话数 ----------
  useEffect(() => {
    if (!user || user === 'anonymous') return;
    api(`/api/sessions?user_id=${encodeURIComponent(user)}&limit=100`, { timeout: 20000 })
      .then((rows) => setSessionCount((rows || []).length))
      .catch(() => setSessionCount(0));
  }, [user, sessionsKey]);

  // ---------- 资料覆盖（顶栏徽标） ----------
  const loadDocsStat = useCallback(async () => {
    try {
      const data = await api('/api/ingest/documents?limit=100', { timeout: 20000 });
      const items = data?.items || [];
      setDocsStat({
        docs: items.length,
        chunks: items.reduce((s, d) => s + (d.chunk_count || 0), 0),
        loading: false,
      });
    } catch {
      setDocsStat({ docs: 0, chunks: 0, loading: false });
    }
  }, []);

  useEffect(() => {
    loadDocsStat();
  }, [loadDocsStat, view]);

  function openSession(id) {
    setSessionId(id);
    setDrawer(null);
    setView('chat');
  }

  function cycleTheme() {
    const i = THEME_STEPS.findIndex((t) => t.v === theme);
    setTheme(THEME_STEPS[(i + 1) % THEME_STEPS.length].v);
  }

  function cycleFont() {
    const i = FONT_STEPS.findIndex((t) => t.v === fontScale);
    setFontScale(FONT_STEPS[(i + 1) % FONT_STEPS.length].v);
  }

  const healthOk = !!health;
  const healthText = !health
    ? '后端未连接'
    : health.llm_configured === false
      ? '资料库正常 · LLM 未配置'
      : '服务正常';
  const healthDetail = health
    ? `数据库：${health.database ? '已连接' : '未连接'} · LLM：${health.llm_configured ? '已配置' : '未配置'} · 模型：${health.model || '-'}`
    : '无法连接后端服务，请确认 uvicorn 已启动';
  const themeIcon = THEME_STEPS.find((t) => t.v === theme)?.icon || '◐';

  return (
    <>
      {/* 背景层：渐变辉光 + 低透明度模糊散景图（固定，不随滚动） */}
      <div className="app-bg" aria-hidden="true" />

      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="logo">🧯</div>
            <div className="brand-text">
              <div className="title">工厂消防知识问答</div>
              <div className="subtitle">依据法规与预案作答 · 逐条标注出处</div>
            </div>
          </div>

          <button
            type="button"
            className={`docs-badge${view === 'docs' ? ' on' : ''}`}
            onClick={() => setView(view === 'docs' ? 'chat' : 'docs')}
            title="查看 / 管理资料库"
          >
            <span className="docs-badge-icon">📚</span>
            {docsStat.loading ? (
              <span className="docs-badge-text">读取中…</span>
            ) : (
              <>
                <span className="docs-badge-text docs-badge-full">
                  {docsStat.docs} 份资料 · {docsStat.chunks} 块
                </span>
                <span className="docs-badge-text docs-badge-short">
                  {docsStat.docs} · {docsStat.chunks}
                </span>
              </>
            )}
          </button>

          <span className="spacer" />

          <span className="status" title={healthDetail}>
            <i className={`dot ${healthOk ? 'ok' : 'bad'}`} />
            <span className="txt">{healthText}</span>
          </span>

          <div className="topbar-actions">
            <button
              type="button"
              className={`icon-btn${bgOn ? '' : ' on'}`}
              onClick={() => setBgOn((v) => !v)}
              title={bgOn ? '关闭背景图（仅保留渐变）' : '开启背景图'}
            >
              ▧
            </button>
            <button type="button" className="icon-btn" onClick={cycleFont} title={`字号：${FONT_STEPS.find((s) => s.v === fontScale)?.label}（点击切换）`}>
              Aa
            </button>
            <button type="button" className="icon-btn" onClick={cycleTheme} title={`主题：${THEME_STEPS.find((t) => t.v === theme)?.label}（点击切换）`}>
              {themeIcon}
            </button>
            <button
              type="button"
              className={`icon-btn wide${drawer === 'sessions' ? ' on' : ''}`}
              onClick={() => setDrawer(drawer === 'sessions' ? null : 'sessions')}
              title="会话记录"
            >
              💬 <span className="icon-label">会话</span>
              {sessionCount > 0 && <span className="cnt">{sessionCount}</span>}
            </button>
            <button
              type="button"
              className={`icon-btn${drawer === 'settings' ? ' on' : ''}`}
              onClick={() => setDrawer(drawer === 'settings' ? null : 'settings')}
              title="设置"
            >
              ⚙
            </button>
          </div>
        </div>
      </header>

      {view === 'chat' && (
        <ChatTab
          settings={settings}
          user={user}
          sessionId={sessionId}
          setSessionId={setSessionId}
          toast={toast}
          onTurnDone={() => setSessionsKey((k) => k + 1)}
        />
      )}

      {view === 'docs' && (
        <main className="docs-view">
          <div className="view-head">
            <button type="button" className="btn ghost" onClick={() => setView('chat')}>
              ← 返回问答
            </button>
            <span className="view-hint">
              当前覆盖 {docsStat.docs} 份资料 · {docsStat.chunks} 块（导入后会重建检索索引）
            </span>
          </div>
          <DocsTab toast={toast} />
        </main>
      )}

      {drawer === 'sessions' && (
        <Drawer
          side="left"
          title="会话记录"
          subtitle={`当前用户共 ${sessionCount} 个会话`}
          onClose={() => setDrawer(null)}
        >
          <div className="drawer-sessions">
            <SessionsTab
              user={user}
              activeId={sessionId}
              onOpen={openSession}
              toast={toast}
              refreshKey={sessionsKey}
            />
          </div>
        </Drawer>
      )}

      {drawer === 'settings' && (
        <Drawer side="right" title="设置" subtitle="检索参数与本地偏好" onClose={() => setDrawer(null)}>
          <SettingsTab
            settings={settings}
            setSettings={setSettings}
            health={health}
            toast={toast}
            adminToken={adminToken}
            setAdminToken={setAdminToken}
          />
        </Drawer>
      )}

      <div className="toasts">
        {toasts.map((t) => (
          <div key={t.id} className={`toast ${t.kind === 'ok' ? 'ok' : t.kind === 'err' ? 'err' : ''}`}>
            {t.message}
          </div>
        ))}
      </div>
    </>
  );
}
