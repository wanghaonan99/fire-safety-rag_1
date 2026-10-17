'use client';

/** 资料库 Tab：拖拽导入、进度、文档状态、切块预览、删除、重建索引。 */

import { useCallback, useEffect, useRef, useState } from 'react';

import { api, apiUpload, formatTime } from '../lib/api';

const CATEGORIES = [
  { value: 'law', label: '法规标准' },
  { value: 'plan', label: '应急预案' },
  { value: 'training', label: '培训资料' },
  { value: 'general', label: '其他资料' },
];

const STATUS = {
  pending: { text: '排队中', cls: '' },
  parsing: { text: '解析中', cls: 'amber' },
  ocr: { text: 'OCR 识别中', cls: 'amber' },
  chunking: { text: '切块中', cls: 'amber' },
  embedding: { text: '向量化中', cls: 'amber' },
  indexed: { text: '已完成', cls: 'green' },
  failed: { text: '失败', cls: 'red' },
};

const IN_PROGRESS = new Set(['pending', 'parsing', 'ocr', 'chunking', 'embedding']);

const ACCEPT = '.pdf,.docx,.html,.htm,.txt,.md,.png,.jpg,.jpeg,.tif,.tiff';

export default function DocsTab({ toast }) {
  const [docs, setDocs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [category, setCategory] = useState('law');
  const [sync, setSync] = useState(true);
  const [over, setOver] = useState(false);
  const [uploads, setUploads] = useState([]); // {name, percent, status, message}
  const [busy, setBusy] = useState('');
  const [preview, setPreview] = useState(null); // {doc, chunks, loading}
  const fileRef = useRef(null);
  const pollRef = useRef(null);

  const load = useCallback(
    async (silent = false) => {
      if (!silent) setLoading(true);
      try {
        const data = await api('/api/ingest/documents?limit=100');
        setDocs(data?.items || []);
      } catch (err) {
        if (!silent) toast?.(`读取资料列表失败：${err.message}`, 'err');
      } finally {
        if (!silent) setLoading(false);
      }
    },
    [toast]
  );

  useEffect(() => {
    load();
  }, [load]);

  // 有文档在处理中时自动轮询状态（OCR/HyPE 可能要几分钟）
  useEffect(() => {
    const pending = docs.some((d) => IN_PROGRESS.has(d.status));
    if (!pending) {
      clearInterval(pollRef.current);
      return undefined;
    }
    pollRef.current = setInterval(() => load(true), 6000);
    return () => clearInterval(pollRef.current);
  }, [docs, load]);

  async function uploadFiles(files) {
    const list = Array.from(files || []);
    if (!list.length) return;

    for (const file of list) {
      const entry = { name: file.name, percent: 0, status: 'uploading', message: '' };
      setUploads((u) => [entry, ...u]);

      const form = new FormData();
      form.append('file', file, file.name);
      form.append('doc_category', category);
      form.append('sync', String(sync));

      try {
        const data = await apiUpload('/api/ingest/upload', form, {
          onProgress: (p) =>
            setUploads((u) => u.map((x) => (x.name === file.name ? { ...x, percent: p } : x))),
        });
        const msg = data?.skipped
          ? '已存在，跳过'
          : data?.chunk_count
            ? `${data.chunk_count} 块`
            : '已提交后台处理';
        setUploads((u) =>
          u.map((x) => (x.name === file.name ? { ...x, percent: 100, status: 'done', message: msg } : x))
        );
        toast?.(`${file.name} 导入完成（${msg}）`, 'ok');
        load(true);
      } catch (err) {
        setUploads((u) =>
          u.map((x) => (x.name === file.name ? { ...x, status: 'error', message: err.message } : x))
        );
        toast?.(`${file.name} 导入失败：${err.message}`, 'err');
      }
    }
  }

  async function showChunks(doc) {
    setPreview({ doc, chunks: [], loading: true });
    try {
      const data = await api(`/api/ingest/document/${doc.id}/chunks?limit=20`);
      setPreview({ doc, chunks: data?.items || [], loading: false });
    } catch (err) {
      toast?.(`读取切块失败：${err.message}`, 'err');
      setPreview(null);
    }
  }

  async function removeDoc(doc) {
    if (!window.confirm(`删除《${doc.title}》及其 ${doc.chunk_count} 个切块与向量？此操作不可撤销。`)) return;
    setBusy(doc.id);
    try {
      await api(`/api/ingest/document/${doc.id}`, { method: 'DELETE' });
      toast?.('已删除', 'ok');
      setDocs((d) => d.filter((x) => x.id !== doc.id));
    } catch (err) {
      toast?.(`删除失败：${err.message}`, 'err');
    } finally {
      setBusy('');
    }
  }

  async function reindex() {
    if (!window.confirm('重新切块并向量化全部资料？已有向量会被覆盖，跨区域数据库上可能较慢。')) return;
    setBusy('reindex');
    try {
      const data = await api('/api/ingest/reindex', { method: 'POST', body: { document_ids: [] } });
      toast?.(`重建完成：${data?.documents ?? 0} 份 / ${data?.chunks ?? 0} 块`, 'ok');
      load(true);
    } catch (err) {
      toast?.(`重建失败：${err.message}`, 'err');
    } finally {
      setBusy('');
    }
  }

  const totalChunks = docs.reduce((sum, d) => sum + (d.chunk_count || 0), 0);

  return (
    <div className="shell">
      {/* ---------- 导入 ---------- */}
      <div className="card">
        <div className="card-head">
          <h3>导入资料</h3>
          <span className="desc">支持 PDF / Word / HTML / 图片；扫描件会自动走 OCR</span>
        </div>
        <div className="card-body">
          <div
            className={`dropzone${over ? ' over' : ''}`}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              uploadFiles(e.dataTransfer.files);
            }}
            onClick={() => fileRef.current?.click()}
          >
            <div className="big">📄</div>
            <div className="t1">把文件拖到这里，或点击选择</div>
            <div className="t2">支持多选 · {ACCEPT.replace(/\./g, '').toUpperCase()}</div>
            <input
              ref={fileRef}
              type="file"
              accept={ACCEPT}
              multiple
              hidden
              onChange={(e) => {
                uploadFiles(e.target.files);
                e.target.value = '';
              }}
            />
          </div>

          <div className="row" style={{ marginTop: 14 }}>
            <div className="grow">
              <label style={{ fontSize: 13, fontWeight: 600, display: 'block', marginBottom: 5 }}>
                资料类别
              </label>
              <select className="select" value={category} onChange={(e) => setCategory(e.target.value)}>
                {CATEGORIES.map((c) => (
                  <option key={c.value} value={c.value}>{c.label}</option>
                ))}
              </select>
            </div>
            <label className="switch" style={{ marginTop: 22 }}>
              <input type="checkbox" checked={sync} onChange={(e) => setSync(e.target.checked)} />
              <span className="track" />
              <span style={{ fontSize: 13.5 }}>同步等待</span>
            </label>
          </div>
          <div className="hint" style={{ marginTop: 10 }}>
            {sync
              ? '同步：请求会一直等到切块与向量化完成（大文件请耐心等，页面不会自动刷新）。'
              : '后台：立即返回，资料状态可在下方列表查看，页面每 6 秒自动刷新。'}
          </div>

          {uploads.length > 0 && (
            <div style={{ marginTop: 14 }}>
              {uploads.map((u, i) => (
                <div className="upload-item" key={`${u.name}-${i}`}>
                  <span className="name">{u.name}</span>
                  {u.status === 'uploading' && <span className="chip blue">上传 {u.percent}%</span>}
                  {u.status === 'done' && <span className="chip green">✓ {u.message}</span>}
                  {u.status === 'error' && <span className="chip red" title={u.message}>✗ 失败</span>}
                </div>
              ))}
              <div className="progress" style={{ display: uploads.some((u) => u.status === 'uploading') ? 'block' : 'none' }}>
                <i style={{ width: `${uploads[0]?.percent || 0}%` }} />
              </div>
              <div className="row" style={{ marginTop: 10 }}>
                <button className="btn sm ghost" onClick={() => setUploads([])}>清空记录</button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ---------- 资料列表 ---------- */}
      <div className="card">
        <div className="card-head">
          <h3>已导入资料</h3>
          <span className="desc">{docs.length} 份 · {totalChunks} 个切块</span>
          <div className="spacer" />
          <button className="btn sm" onClick={() => load()} disabled={loading}>
            {loading ? '刷新中…' : '↻ 刷新'}
          </button>
          <button className="btn sm" onClick={reindex} disabled={busy === 'reindex' || docs.length === 0}>
            {busy === 'reindex' ? '重建中…' : '重建索引'}
          </button>
        </div>

        {loading && docs.length === 0 ? (
          <div className="empty">正在加载…</div>
        ) : docs.length === 0 ? (
          <div className="empty">
            <div className="big">📚</div>
            还没有资料，先导入一份消防法规或预案吧
          </div>
        ) : (
          <div className="list">
            {docs.map((d) => {
              const st = STATUS[d.status] || { text: d.status, cls: '' };
              return (
                <div className="item" key={d.id}>
                  <div className="main">
                    <div className="name" title={d.title}>{d.title}</div>
                    <div className="sub">
                      <span className={`chip ${st.cls}`}>{st.text}</span>
                      <span className="chip">
                        {CATEGORIES.find((c) => c.value === d.doc_category)?.label || d.doc_category}
                      </span>
                      <span>{d.chunk_count} 块</span>
                      {d.page_count > 0 && <span>· {d.page_count} 页</span>}
                      {d.is_scanned && <span className="chip amber">扫描件 OCR</span>}
                      <span>· {formatTime(d.created_at)}</span>
                    </div>
                    {d.error_message && (
                      <div className="sub" style={{ color: 'var(--warn)' }} title={d.error_message}>
                        {d.error_message.slice(0, 120)}
                      </div>
                    )}
                  </div>
                  <div className="actions">
                    <button className="btn sm" onClick={() => showChunks(d)}>查看切块</button>
                    <button className="btn sm danger" disabled={busy === d.id} onClick={() => removeDoc(d)}>
                      {busy === d.id ? '删除中' : '删除'}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* ---------- 切块预览 ---------- */}
      {preview && (
        <div className="modal-mask" onClick={() => setPreview(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-head">
              <h3>《{preview.doc.title}》切块预览</h3>
              <span className="desc">
                {preview.loading ? '加载中…' : `共 ${preview.doc.chunk_count} 块，显示前 ${preview.chunks.length} 块`}
              </span>
              <div className="spacer" />
              <button className="btn sm" onClick={() => setPreview(null)}>关闭</button>
            </div>
            <div className="modal-body">
              {preview.loading ? (
                <div className="empty">正在加载切块…</div>
              ) : (
                preview.chunks.map((c) => (
                  <div className="chunk" key={c.id}>
                    <div className="h">
                      <span className="chip blue">#{c.chunk_index}</span>
                      {c.heading_path && <span>{c.heading_path}</span>}
                      {c.page_no > 0 && <span>第 {c.page_no} 页</span>}
                      <span className="chip">{c.token_count} tk</span>
                    </div>
                    <div className="t">{c.content.slice(0, 500)}{c.content.length > 500 ? '…' : ''}</div>
                    {c.hypothetical_questions?.length > 0 && (
                      <div className="q">
                        <b>HyPE 假想问题：</b>
                        {c.hypothetical_questions.map((q, i) => (
                          <div key={i}>· {q}</div>
                        ))}
                      </div>
                    )}
                  </div>
                ))
              )}
            </div>
            <div className="modal-foot">
              <button className="btn" onClick={() => setPreview(null)}>关闭</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
