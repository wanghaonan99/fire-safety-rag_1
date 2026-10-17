/** 后端 API 封装：统一基址、统一解包 {code,message,data}、统一报错。 */

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

/** 统一的响应解包：后端一律返回 {code,message,data}，出错时抛 Error(detail|message) */
export async function api(path, { method = 'GET', body, headers, token, timeout = 300000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method,
      headers: {
        ...(body instanceof FormData ? {} : { 'Content-Type': 'application/json' }),
        ...(token ? { 'X-Admin-Token': token } : {}),
        ...headers,
      },
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
      signal: controller.signal,
    });
    const text = await res.text();
    let parsed;
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = { message: text };
    }
    if (!res.ok) {
      throw new Error(parsed?.detail || parsed?.message || `HTTP ${res.status}`);
    }
    return parsed.data !== undefined ? parsed.data : parsed;
  } catch (err) {
    if (err.name === 'AbortError') throw new Error(`请求超时（${Math.round(timeout / 1000)} 秒）`);
    throw err;
  } finally {
    clearTimeout(timer);
  }
}

/** 带上传进度的 POST（fetch 拿不到上传进度，只能用 XHR） */
export function apiUpload(path, formData, { token, onProgress } = {}) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_BASE}${path}`);
    if (token) xhr.setRequestHeader('X-Admin-Token', token);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      let parsed;
      try {
        parsed = JSON.parse(xhr.responseText);
      } catch {
        parsed = { message: xhr.responseText };
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(parsed.data ?? parsed);
      else reject(new Error(parsed?.detail || parsed?.message || `HTTP ${xhr.status}`));
    };
    xhr.onerror = () => reject(new Error('网络错误（后端是否在运行？）'));
    xhr.timeout = 30 * 60 * 1000; // 扫描件 OCR + HyPE 可能很久
    xhr.ontimeout = () => reject(new Error('上传超时'));
    xhr.send(formData);
  });
}

/* ---------------- localStorage 小工具（都做异常兜底，隐私模式下不崩） ---------------- */

export function loadLocal(key, fallback) {
  if (typeof window === 'undefined') return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch {
    return fallback;
  }
}

export function saveLocal(key, value) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* 忽略 */
  }
}

/** 稳定的匿名用户标识：用于把会话按浏览器隔离 */
export function getUserId() {
  if (typeof window === 'undefined') return 'anonymous';
  let uid = loadLocal('fsrag.user_id', null);
  if (!uid) {
    uid = `web_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36).slice(-4)}`;
    saveLocal('fsrag.user_id', uid);
  }
  return uid;
}

export function formatTime(iso) {
  if (!iso) return '';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '';
  const diff = (Date.now() - d.getTime()) / 1000;
  if (diff < 60) return '刚刚';
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} 天前`;
  return d.toLocaleDateString('zh-CN');
}
