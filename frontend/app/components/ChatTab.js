'use client';

/**
 * 问答 Tab：对话流 + 结构化答案 + 可点引用 + 出处面板。
 *
 * 这一版重点解决四个"影响可用性"的问题：
 *   1. 答案里的 Markdown 表格原来显示成原始竖线 → 现在渲染成真表格；
 *   2. 引用 [n] 原来是静态文字 → 现在可点，右侧出处面板定位并高亮；
 *   3. 拒答（资料中没有）原来和正常回答同样式 → 现在用琥珀色"未收录"样式；
 *   4. 出处原来默认折叠 → 现在引用条常驻在答案下方。
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { api } from '../lib/api';
import { detectRefusal, parseAnswer, toPlainText } from '../lib/renderAnswer';

const SUGGESTIONS = [
  '消防安全重点单位多久组织演练一次？',
  '甲、乙类厂房与相邻工厂的防火间距是多少？',
  '应急预案应当包括哪些内容？',
  '违规占用消防车通道会受到什么处罚？',
];

/** 演示数据：访问 ?demo=1 时直接渲染，用来快速看渲染效果（不请求后端）。 */
const DEMO_TURNS = [
  { role: 'user', content: '我们厂有甲B类液体汽车装卸鹤管，它到厂房（生产设施）的防火间距最少要留多少米？' },
  {
    role: 'bot',
    content: `- 结论：应按表 4.1.5 对照取值，其中"甲、乙类厂房（生产设施）"一列对相邻工厂为 30m。
- 依据：
  - "精细化工企业与相邻工厂或设施的防火间距不应小于表4.1.5的规定。"[1]
  - "相邻工厂指除精细化工企业以外的不同类工厂。若相邻工厂有相关的国家标准规定时，应从严执行。"[1]
- 操作要点：
  1. 先确认相邻对象是否属于"精细化工企业"：属于则按表 4.1.6[2]，不属于按表 4.1.5[1]。
  2. 按设施类别与储罐容积对照下表取值。
  3. 涉及丙类可燃液体时按注 8 的折减规则执行。

| 相邻工厂或设施 | 液化烃储罐 | 甲、乙类液体储罐 | 甲、乙类厂房 | 全厂性重要设施 |
| --- | --- | --- | --- | --- |
| 相邻工厂（围墙或边界线） | 27 / 30 / 35 | 30 / 35 | 30 | 40 |
| 居住区、村镇及重要公共建筑 | 90 / 100 / 140 | 50 / 60 | 50 | 25 |`,
    sources: [
      {
        document_title: '精细化工企业工程设计防火标准（GB 51283-2020）',
        heading_path: '4.1.5 精细化工企业与相邻工厂或设施的防火间距',
        page_no: 14,
        score: 7.52,
        content:
          '4.1.5 精细化工企业与相邻工厂或设施的防火间距不应小于表4.1.5的规定。注: 1 居住区、村镇指1000人或300户及以上者；2 相邻工厂指除精细化工企业以外的不同类工厂。若相邻工厂有相关的国家标准规定时，应从严执行；3 分母为与高层民用建筑的防火间距，分子为与其它建筑的防火间距。',
      },
      {
        document_title: '精细化工企业工程设计防火标准（GB 51283-2020）',
        heading_path: '4.1.6 相邻精细化工企业之间的防火间距',
        page_no: 15,
        score: 6.53,
        content: '4.1.6 相邻精细化工企业的防火间距不应小于表4.1.6的规定。本条对精细化工园区（基地）内企业之间的防火间距规定，是根据产业集聚、精细化…',
      },
    ],
    latency: 7500,
    cached: false,
    demo: true,
  },
  { role: 'user', content: '我们厂喷头多久换一次？' },
  {
    role: 'bot',
    content: `- 结论：根据现有资料无法确定喷头更换周期的具体要求。
- 依据：参考资料仅提到自动喷水灭火系统的设计应符合《自动喷水灭火系统设计规范》GB 50084[1]，未涉及喷头更换周期。
- 操作要点：建议查阅 GB 50084 原文或咨询消防维保单位。`,
    sources: [
      {
        document_title: '精细化工企业工程设计防火标准（GB 51283-2020）',
        heading_path: '9.4 自动灭火系统',
        page_no: 46,
        score: -0.24,
        content: '9.4.1 自动喷水灭火系统的设计应符合现行国家标准《自动喷水灭火系统设计规范》GB 50084的有关规定。',
      },
    ],
    latency: 6200,
    cached: false,
    demo: true,
  },
];

function isDemoMode() {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('demo') === '1';
}

/** 行内渲染：加粗 / 行内代码 / 可点引用。 */
function Inline({ tokens, onCite }) {
  return tokens.map((tk, i) => {
    if (tk.t === 'strong') return <strong key={i}>{tk.v}</strong>;
    if (tk.t === 'code') return <code key={i} className="ans-code">{tk.v}</code>;
    if (tk.t === 'cite') {
      return (
        <button
          key={i}
          type="button"
          className="cite-btn"
          onClick={() => onCite?.(tk.n)}
          title={`查看第 ${tk.n} 条出处`}
        >
          {tk.v}
        </button>
      );
    }
    return <span key={i}>{tk.v}</span>;
  });
}

const SECTION_LABEL = { conclusion: '结论', basis: '依据', action: '怎么做' };

/**
 * 结构化答案：按"段"分组渲染。
 *   结论 → 独立高亮块（暖色底 + 左侧竖条 + 加大字号）—— 让人一眼抓住答案
 *   依据 → 逐条引文卡（文档/条款为小标题，原文为引文主体）
 *   怎么做 → 带序号的步骤列表（纯定义题通常没有这一段）
 */
function AnswerBlocks({ text, onCite }) {
  const sections = useMemo(() => {
    const blocks = parseAnswer(text);
    const out = [];
    let cur = { kind: 'none', blocks: [] };
    for (const b of blocks) {
      if (b.type === 'heading') {
        if (cur.kind !== 'none' || cur.blocks.length) out.push(cur);
        cur = { kind: b.kind, blocks: [] };
      } else {
        cur.blocks.push(b);
      }
    }
    if (cur.kind !== 'none' || cur.blocks.length) out.push(cur);
    return out;
  }, [text]);

  const renderBlock = (b, key) => {
    if (b.type === 'basisItem') {
      return (
        <div key={key} className="basis-item">
          {(b.doc || b.clause) && (
            <div className="basis-src">
              {b.doc ? <span className="basis-doc">{b.doc}</span> : null}
              {b.clause ? <span className="basis-clause">{b.clause}</span> : null}
            </div>
          )}
          <div className="basis-text">
            <Inline tokens={b.inline} onCite={onCite} />
          </div>
        </div>
      );
    }
    if (b.type === 'list') {
      const Tag = b.ordered ? 'ol' : 'ul';
      return (
        <Tag key={key} className="ans-list">
          {b.items.map((it, k) => (
            <li key={k}>
              <Inline tokens={it} onCite={onCite} />
            </li>
          ))}
        </Tag>
      );
    }
    if (b.type === 'table') {
      return (
        <div key={key} className="ans-table-wrap">
          <table className="ans-table">
            <thead>
              <tr>
                {b.header.map((h, k) => (
                  <th key={k}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {b.rows.map((r, k) => (
                <tr key={k}>
                  {r.map((c, m) => (
                    <td key={m}>{c}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      );
    }
    return (
      <p key={key} className="para">
        <Inline tokens={b.inline} onCite={onCite} />
      </p>
    );
  };

  return (
    <div className="answer">
      {sections.map((s, si) => (
        <section key={si} className={`ans-sec ans-sec-${s.kind}`}>
          {s.kind !== 'none' && (
            <div className={`ans-h ans-h-${s.kind}`}>
              <span className="ans-h-dot" />
              {SECTION_LABEL[s.kind] || s.kind}
            </div>
          )}
          {s.blocks.map((b, i) => renderBlock(b, `${si}-${i}`))}
        </section>
      ))}
    </div>
  );
}

/** 引用条：常驻在答案下方（不再默认折叠）。 */
function CiteBar({ sources, onOpen }) {
  if (!sources?.length) return null;
  return (
    <div className="cite-bar">
      <span className="cite-bar-label">出处</span>
      {sources.map((s, i) => (
        <button key={i} type="button" className="cite-chip" onClick={() => onOpen(i + 1)}>
          <b>{i + 1}</b>
          <span className="cite-chip-doc">{s.document_title || '未命名'}</span>
          {s.heading_path ? <span className="cite-chip-path">{s.heading_path}</span> : null}
        </button>
      ))}
    </div>
  );
}

/** 右侧出处面板：点引用后定位并高亮对应条目。 */
function SourcePanel({ sources, focus, onClose }) {
  const refs = useRef([]);

  useEffect(() => {
    const node = refs.current[focus - 1];
    if (node) node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [focus, sources]);

  return (
    <aside className="src-panel">
      <div className="src-panel-head">
        <span className="src-panel-title">出处原文 · {sources.length} 条</span>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="关闭">
          ✕
        </button>
      </div>
      <div className="src-list">
        {sources.map((s, i) => (
          <div
            key={i}
            ref={(el) => {
              refs.current[i] = el;
            }}
            className={`src${focus === i + 1 ? ' focus' : ''}`}
          >
            <div className="src-head">
              <span className="src-idx">{i + 1}</span>
              <span className="src-doc">《{s.document_title || '未命名'}》</span>
              {s.page_no > 0 && <span className="src-path">第 {s.page_no} 页</span>}
              <span className={s.score >= 0 ? 'score' : 'score low'}>
                相关度 {Number(s.score || 0).toFixed(2)}
              </span>
            </div>
            {s.heading_path && <div className="src-path src-panel-path">{s.heading_path}</div>}
            <div className="src-text">{s.content}</div>
          </div>
        ))}
      </div>
    </aside>
  );
}

export default function ChatTab({ settings, user, sessionId, setSessionId, toast, onTurnDone }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [panel, setPanel] = useState(null); // { msgIdx, focus }
  const inputRef = useRef(null);
  const threadRef = useRef(null);

  // 演示模式（?demo=1）：不请求后端，直接看渲染效果
  useEffect(() => {
    if (isDemoMode()) setMessages(DEMO_TURNS);
  }, []);

  // 从"会话"页点进来时，加载该会话的历史
  useEffect(() => {
    let alive = true;
    if (!sessionId) {
      if (!isDemoMode()) setMessages([]);
      return () => {
        alive = false;
      };
    }
    setLoadingHistory(true);
    api(`/api/sessions/${sessionId}/history`)
      .then((rows) => {
        if (!alive) return;
        setMessages((rows || []).map((r) => ({ role: r.role === 'user' ? 'user' : 'bot', content: r.content })));
      })
      .catch(() => alive && setMessages([]))
      .finally(() => alive && setLoadingHistory(false));
    return () => {
      alive = false;
    };
  }, [sessionId]);

  useEffect(() => {
    if (!loading) return undefined;
    setElapsed(0);
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [loading]);

  useEffect(() => {
    const el = threadRef.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
  }, [messages, loading]);

  // 桌面端自动聚焦；小屏不聚焦（否则一进页面就弹键盘，挡住半个屏幕）
  useEffect(() => {
    if (typeof window !== 'undefined' && window.innerWidth >= 900) {
      inputRef.current?.focus();
    }
  }, []);

  // 键盘：/ 聚焦输入，Esc 关面板，↑ 调出上一个问题（输入框为空时）
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') setPanel(null);
      if (e.key === '/' && document.activeElement !== inputRef.current) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (
        e.key === 'ArrowUp' &&
        document.activeElement === inputRef.current &&
        !input.trim()
      ) {
        const lastUser = [...messages].reverse().find((m) => m.role === 'user');
        if (lastUser) {
          e.preventDefault();
          setInput(lastUser.content);
        }
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [messages, input]);

  const copy = useCallback(
    async (text, label) => {
      try {
        await navigator.clipboard.writeText(text);
        toast?.(`${label}已复制`, 'ok');
      } catch {
        toast?.('复制失败，请手动选择文本', 'err');
      }
    },
    [toast]
  );

  const ask = useCallback(
    async (raw, opts = {}) => {
      const question = (raw ?? input).trim();
      if (!question || loading) return;

      if (!opts.silentUser) setMessages((m) => [...m, { role: 'user', content: question }]);
      setInput('');
      setLoading(true);

      try {
        const d = await api('/api/query', {
          method: 'POST',
          body: {
            question,
            session_id: sessionId,
            user_id: user,
            top_k: settings.topK,
            use_cache: settings.useCache,
            use_query_transform: settings.useQueryTransform,
            doc_category: settings.docCategory || null,
          },
          timeout: 600000, // 首次要加载模型，给足 10 分钟
        });
        if (d.session_id && d.session_id !== sessionId) setSessionId(d.session_id);
        setMessages((m) => [
          ...m,
          {
            role: 'bot',
            content: d.answer || '（没有返回内容）',
            sources: d.sources || [],
            cached: d.cached,
            latency: d.latency_ms,
            transformed: d.transformed_queries || [],
          },
        ]);
        onTurnDone?.();
      } catch (err) {
        setMessages((m) => [
          ...m,
          { role: 'bot', content: `请求失败：${err.message}`, error: true, sources: [], retry: question },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [input, loading, sessionId, settings, user, onTurnDone, setSessionId]
  );

  const clearLocal = () => {
    setMessages([]);
    setSessionId(null);
    setPanel(null);
    inputRef.current?.focus();
  };

  // 等待文案分级：让 7.5 秒不再"死等"
  const waitText =
    elapsed < 1 ? '正在检索资料…' : elapsed < 3 ? '正在比对条款…' : `正在组织回答…（${elapsed}s）`;

  return (
    <>
      <div className={`chat-layout${panel ? ' with-panel' : ''}`}>
        <main className="shell chat" ref={threadRef}>
          {messages.length === 0 && !loadingHistory ? (
            <div className="hero">
              <div className="hero-banner">
                {/* 真实消防员合影 + 深色渐变遮罩：给首屏一个"任务开始"的视觉锚点 */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src="/bg/hero.jpg" alt="" />
                <span className="hero-banner-tag">知识库已就绪</span>
                <div className="hero-banner-text">
                  <h2>有什么消防问题想确认？</h2>
                  <p>
                    回答仅依据已导入的法规与预案，逐条标注出处；资料中没有的内容会明确说明，不会编造。
                  </p>
                </div>
              </div>
              <div className="chips-wrap">
                {SUGGESTIONS.map((s) => (
                  <button className="chip-btn" key={s} onClick={() => ask(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread">
              {loadingHistory && (
                <div className="loading">
                  <span className="dots">
                    <i />
                    <i />
                    <i />
                  </span>
                  正在加载历史…
                </div>
              )}
              {messages.map((m, i) => {
                const refusal = m.role === 'bot' && !m.error && detectRefusal(m.content);
                return (
                  <div className={`msg ${m.role === 'user' ? 'user' : 'bot'}`} key={i}>
                    <div className={`avatar ${m.role === 'user' ? 'me' : 'bot'}`}>
                      {m.role === 'user' ? '我' : '🔥'}
                    </div>
                    <div className={`bubble${m.error ? ' err' : ''}${refusal ? ' notice' : ''}`}>
                      {m.role === 'user' ? (
                        m.content
                      ) : (
                        <>
                          {refusal && (
                            <div className="refusal-banner">
                              <span className="refusal-icon">⚠</span>
                              资料中未收录相关内容 · 以下说明不是规范原文
                            </div>
                          )}
                          {m.error ? (
                            <div className="err-body">
                              <div>{m.content}</div>
                              {m.retry && (
                                <button
                                  type="button"
                                  className="btn ghost err-retry"
                                  onClick={() => {
                                    setMessages((list) => list.filter((_, k) => k !== i));
                                    ask(m.retry, { silentUser: true });
                                  }}
                                >
                                  重试
                                </button>
                              )}
                            </div>
                          ) : (
                            <AnswerBlocks
                              text={m.content}
                              onCite={(n) => setPanel({ msgIdx: i, focus: n })}
                            />
                          )}
                        </>
                      )}

                      {m.role === 'bot' && !m.error && (
                        <>
                          <CiteBar sources={m.sources} onOpen={(n) => setPanel({ msgIdx: i, focus: n })} />
                          <div className="meta-row">
                            {m.cached && <span className="chip blue">语义缓存命中</span>}
                            {m.latency ? <span className="chip">{(m.latency / 1000).toFixed(1)} 秒</span> : null}
                            {m.transformed?.length ? (
                              <span className="chip amber" title={m.transformed.join(' / ')}>
                                已查询改写
                              </span>
                            ) : null}
                            <span className="chip">deepseek-chat</span>
                            <button
                              type="button"
                              className="chip chip-btn-sm"
                              onClick={() => copy(toPlainText(m.content), '答案')}
                            >
                              复制答案
                            </button>
                            {m.sources?.length ? (
                              <button
                                type="button"
                                className="chip chip-btn-sm"
                                onClick={() =>
                                  copy(
                                    m.sources
                                      .map(
                                        (s, k) =>
                                          `[${k + 1}]《${s.document_title}》${s.heading_path || ''}${
                                            s.page_no ? ` 第${s.page_no}页` : ''
                                          }\n${s.content}`
                                      )
                                      .join('\n\n'),
                                    '出处'
                                  )
                                }
                              >
                                复制出处
                              </button>
                            ) : null}
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                );
              })}

              {loading && (
                <div className="msg bot">
                  <div className="avatar bot">🔥</div>
                  <div className="bubble">
                    <div className="loading">
                      <span className="dots">
                        <i />
                        <i />
                        <i />
                      </span>
                      {waitText}
                    </div>
                    <div className="progress-line">
                      <i style={{ width: `${Math.min(96, 8 + elapsed * 7)}%` }} />
                    </div>
                    {elapsed > 12 && (
                      <div className="hint">
                        首次提问需要把向量模型与精排模型加载进显存，通常 60~90 秒；之后的提问约 8 秒。
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </main>

        {panel && messages[panel.msgIdx]?.sources?.length ? (
          <SourcePanel
            sources={messages[panel.msgIdx].sources}
            focus={panel.focus}
            onClose={() => setPanel(null)}
          />
        ) : null}
      </div>

      <div className="composer">
        <form
          className="composer-inner"
          onSubmit={(e) => {
            e.preventDefault();
            ask();
          }}
        >
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="例如：甲、乙类厂房与相邻工厂的防火间距是多少？"
            enterKeyHint="send"
            autoComplete="off"
          />
          {messages.length > 0 && (
            <button
              type="button"
              className="btn ghost new-session"
              onClick={clearLocal}
              title="新建会话（保留历史记录）"
            >
              ＋<span className="new-session-label">新会话</span>
            </button>
          )}
          <button className="send" type="submit" disabled={loading || !input.trim()}>
            {loading ? '生成中' : '发送'}
          </button>
        </form>
        <div className="foot">
          <span className="foot-session">{sessionId ? `当前会话 ${sessionId.slice(0, 8)}… · ` : ''}</span>
          <span className="foot-keys">Enter 发送 · Shift+Enter 换行 · 按 / 聚焦输入 · Esc 关闭出处 · </span>
          <span className="foot-note">回答仅供参考，请以规范原文为准</span>
        </div>
      </div>
    </>
  );
}
