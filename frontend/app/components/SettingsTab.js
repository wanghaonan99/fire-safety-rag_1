'use client';

/** 设置 Tab：检索参数、管理接口（服务配置/统计/清缓存）、健康状态。 */

import { useCallback, useEffect, useState } from 'react';

import { api, API_BASE, saveLocal } from '../lib/api';

const CATEGORIES = [
  { value: '', label: '全部资料' },
  { value: 'law', label: '仅法规标准' },
  { value: 'plan', label: '仅应急预案' },
  { value: 'training', label: '仅培训资料' },
  { value: 'general', label: '仅其他资料' },
];

export default function SettingsTab({ settings, setSettings, health, toast, adminToken, setAdminToken }) {
  const [tokenInput, setTokenInput] = useState(adminToken || '');
  const [config, setConfig] = useState(null);
  const [stats, setStats] = useState(null);
  const [busy, setBusy] = useState('');

  useEffect(() => {
    setTokenInput(adminToken || '');
  }, [adminToken]);

  const update = useCallback(
    (patch) => {
      const next = { ...settings, ...patch };
      setSettings(next);
      saveLocal('fsrag.settings', next);
    },
    [settings, setSettings]
  );

  function saveToken() {
    const value = tokenInput.trim();
    // HTTP 头不允许非 ASCII，先在这里拦掉，避免 fetch 抛难懂的编码错误
    if (value && !/^[\x20-\x7E]+$/.test(value)) {
      toast?.('口令含非 ASCII 字符，HTTP 头不允许。请用 secrets.token_urlsafe(32) 生成的值', 'err');
      return;
    }
    setAdminToken(value);
    saveLocal('fsrag.admin_token', value);
    toast?.(value ? '管理口令已保存到本机浏览器' : '已清除管理口令', 'ok');
  }

  async function callAdmin(path, label, method = 'GET') {
    if (!adminToken) {
      toast?.('请先填写管理口令（.env 里的 ADMIN_TOKEN）', 'err');
      return null;
    }
    setBusy(label);
    try {
      const data = await api(path, { method, token: adminToken, timeout: 60000 });
      toast?.(`${label}成功`, 'ok');
      return data;
    } catch (err) {
      const msg = err.message?.includes('401') ? '口令不正确（检查后端 .env 的 ADMIN_TOKEN）' : err.message;
      toast?.(`${label}失败：${msg}`, 'err');
      return null;
    } finally {
      setBusy('');
    }
  }

  return (
    <div className="shell">
      {/* ---------- 检索参数 ---------- */}
      <div className="card">
        <div className="card-head">
          <h3>检索参数</h3>
          <span className="desc">只影响你本机浏览器发出的提问</span>
        </div>
        <div className="card-body">
          <div className="row">
            <div className="grow">
              <div className="field">
                <label>送入 LLM 的片段数（top_k）</label>
                <input
                  className="input"
                  type="number"
                  min={1}
                  max={20}
                  value={settings.topK}
                  onChange={(e) => update({ topK: Math.max(1, Math.min(20, Number(e.target.value) || 1)) })}
                />
                <div className="hint">越大上下文越全，但生成越慢。默认 5</div>
              </div>
            </div>
            <div className="grow">
              <div className="field">
                <label>资料范围</label>
                <select
                  className="select"
                  value={settings.docCategory}
                  onChange={(e) => update({ docCategory: e.target.value })}
                >
                  {CATEGORIES.map((c) => (
                    <option key={c.value} value={c.value}>{c.label}</option>
                  ))}
                </select>
                <div className="hint">按资料类别过滤检索范围（向量与 BM25 都会过滤）</div>
              </div>
            </div>
          </div>

          <div className="row" style={{ gap: 26 }}>
            <label className="switch">
              <input
                type="checkbox"
                checked={settings.useCache}
                onChange={(e) => update({ useCache: e.target.checked })}
              />
              <span className="track" />
              <span style={{ fontSize: 13.5 }}>语义缓存（相同/近义问题秒回）</span>
            </label>
            <label className="switch">
              <input
                type="checkbox"
                checked={settings.useQueryTransform}
                onChange={(e) => update({ useQueryTransform: e.target.checked })}
              />
              <span className="track" />
              <span style={{ fontSize: 13.5 }}>Step-back 查询改写</span>
            </label>
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            Step-back 会先让 LLM 生成一个更上位的问题、再检索一遍：召回可能更好，但
            在跨区域数据库上每次要多花 5~15 秒。默认关闭。
          </div>
        </div>
      </div>

      {/* ---------- 服务状态 ---------- */}
      <div className="card">
        <div className="card-head">
          <h3>服务状态</h3>
          <span className="desc">{API_BASE}</span>
          <div className="spacer" />
          <span className={`chip ${health?.status === 'ok' ? 'green' : 'red'}`}>
            {health?.status === 'ok' ? '运行中' : '未连接'}
          </span>
        </div>
        <div className="card-body">
          {health ? (
            <>
              <div className="kv"><span className="k">应用 / 环境</span><span className="v">{health.app} · {health.environment}</span></div>
              <div className="kv"><span className="k">数据库</span><span className="v">{health.database ? '✓ 已连接' : '✗ 未连接'}</span></div>
              <div className="kv"><span className="k">语义缓存后端</span><span className="v">{health.cache_backend || '—'}{health.redis ? '（Redis）' : '（内存 LRU）'}</span></div>
              <div className="kv"><span className="k">LLM 密钥</span><span className="v">{health.llm_configured ? '✓ 已配置' : '✗ 未配置'}</span></div>
              <div className="kv"><span className="k">版本</span><span className="v">{health.version}</span></div>
            </>
          ) : (
            <div className="empty" style={{ padding: 20 }}>
              后端未连接。请确认已启动：<code>uvicorn app.main:app --reload --port 8000</code>
            </div>
          )}
        </div>
      </div>

      {/* ---------- 管理接口 ---------- */}
      <div className="card">
        <div className="card-head">
          <h3>管理接口</h3>
          <span className="desc">需要后端 .env 里的 ADMIN_TOKEN</span>
        </div>
        <div className="card-body">
          <div className="field">
            <label>管理口令</label>
            <div className="row">
              <input
                className="input grow"
                type="password"
                placeholder="粘贴 .env 中 ADMIN_TOKEN 的值"
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
              />
              <button className="btn" onClick={saveToken}>保存到浏览器</button>
            </div>
            <div className="hint">
              只存在本机 localStorage，不会上传。生成方法：
              <code>python -c &quot;import secrets;print(secrets.token_urlsafe(32))&quot;</code>
            </div>
          </div>

          <div className="row">
            <button
              className="btn"
              disabled={!!busy}
              onClick={async () => setConfig(await callAdmin('/api/admin/config', '读取配置'))}
            >
              {busy === '读取配置' ? '读取中…' : '读取服务配置'}
            </button>
            <button
              className="btn"
              disabled={!!busy}
              onClick={async () => setStats(await callAdmin('/api/admin/stats', '读取统计'))}
            >
              {busy === '读取统计' ? '读取中…' : '知识库统计'}
            </button>
            <button
              className="btn danger"
              disabled={!!busy}
              onClick={async () => {
                if (!window.confirm('清空语义缓存？下次提问会重新检索与生成（不影响知识库数据）。')) return;
                const d = await callAdmin('/api/admin/cache/clear', '清空缓存', 'POST');
                if (d) toast?.(`已删除 ${d.removed ?? 0} 条缓存`, 'ok');
              }}
            >
              {busy === '清空缓存' ? '清理中…' : '清空语义缓存'}
            </button>
          </div>

          {config && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>服务端生效配置</div>
              {Object.entries(config).map(([k, v]) => (
                <div className="kv" key={k}>
                  <span className="k">{k}</span>
                  <span className="v">{String(v)}</span>
                </div>
              ))}
            </div>
          )}

          {stats && (
            <div style={{ marginTop: 16 }}>
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6 }}>知识库统计</div>
              {Object.entries(stats).map(([k, v]) => (
                <div className="kv" key={k}>
                  <span className="k">{k}</span>
                  <span className="v">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-head"><h3>关于</h3></div>
        <div className="card-body" style={{ fontSize: 13, color: 'var(--ink-2)' }}>
          <p style={{ marginTop: 0 }}>
            工厂消防知识问答助手 · 第一批（混合检索 + Rerank + 评估基线）+ 第二批（OCR、HyPE、查询改写、语义缓存）。
          </p>
          <p style={{ marginBottom: 0 }}>
            技术栈：FastAPI · LlamaIndex · pgvector(Neon) · PaddleOCR · MinerU · DeepSeek · bge-m3 · bge-reranker-v2-m3。
            回答仅供参考，请以规范原文为准。
          </p>
        </div>
      </div>
    </div>
  );
}
