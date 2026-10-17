/**
 * 把模型输出的答案解析成可渲染的结构块。
 *
 * 输出结构（由 SYSTEM_PROMPT 约束，属可控子集）：
 *   结论：…（1~2 句，关键数值加粗）
 *   依据：
 *     - 《文档名》条款号：原文引述 [n]
 *   怎么做：（仅操作类问题才有；纯定义题整段省略）
 *
 * 这里把"依据"逐条拆成引文卡，便于前端做出条理感；
 * 无法识别的行**一律回退成普通段落**，绝不吞内容。
 *
 * 块类型：
 *   { type:'heading', kind:'conclusion'|'basis'|'action', text }
 *   { type:'paragraph', sec, inline:[...] }
 *   { type:'list', sec, ordered, items:[[inline...]] }
 *   { type:'basisItem', doc, clause, inline }          ← 依据里的单条引文
 *   { type:'table', header:[string], rows:[[string]] }
 *
 * inline 令牌：{ t:'text'|'strong'|'code'|'cite', v:string, n?:number }
 */

const SECTION_TITLES = [
  { kind: 'conclusion', re: /^[\s\-*•·>#]*结论[:：]\s*/, label: '结论' },
  { kind: 'basis', re: /^[\s\-*•·>#]*依据[:：]\s*/, label: '依据' },
  { kind: 'action', re: /^[\s\-*•·>#]*(怎么做|操作要点|落地做法|执行步骤|处理方法)[:：]\s*/, label: '怎么做' },
];

const BULLET_RE = /^[\s]*[-*•·]\s+/;
const ORDERED_RE = /^[\s]*(\d+)[.、)]\s+/;

/** 拒答判定：只看开头，避免正文里的"无法确定"误判。 */
export function detectRefusal(text = '') {
  const head = text.slice(0, 200);
  return /根据现有资料无法确定|资料中未(找到|提供|收录|包含|明确|规定)|未收录|无相关(内容|规定)/.test(head);
}

/** 行内解析：加粗 / 行内代码 / [n] 引用。 */
export function tokenizeInline(text = '') {
  const out = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g;
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ t: 'text', v: text.slice(last, m.index) });
    const s = m[0];
    if (s.startsWith('**')) out.push({ t: 'strong', v: s.slice(2, -2) });
    else if (s.startsWith('`')) out.push({ t: 'code', v: s.slice(1, -1) });
    else out.push({ t: 'cite', v: s, n: Number(s.slice(1, -1)) });
    last = m.index + s.length;
  }
  if (last < text.length) out.push({ t: 'text', v: text.slice(last) });
  return out;
}

/**
 * 解析"依据"里的单条引文，支持三种常见形态：
 *   - 《GB 51283》4.1.5：原文"…"[1]
 *   - 《GB 51283》4.1.5 原文"…"[1]
 *   - GB 51283 4.1.5：原文"…"[1]      （模型偶尔漏书名号）
 */
function parseBasisLine(line) {
  const withBook = /^[-*•·\s]*《([^》]{2,80})》\s*([^：:]{0,50})?[：:]\s*([\s\S]+)$/.exec(line);
  if (withBook) {
    return { doc: `《${withBook[1].trim()}》`, clause: (withBook[2] || '').trim(), text: withBook[3].trim() };
  }
  const noBook = /^[-*•·\s]*([^《：:]{2,40}?(?:[》条章节]|\d+(?:\.\d+)+))\s*[：:]\s*([\s\S]+)$/.exec(line);
  if (noBook) {
    return { doc: '', clause: noBook[1].trim(), text: noBook[2].trim() };
  }
  return null;
}

function isSepRow(line) {
  return /^\s*\|?[\s:|-]+\|[\s:|-]*\|?\s*$/.test(line) && line.includes('|') && line.includes('-');
}

function splitRow(line) {
  return line
    .trim()
    .replace(/^\|/, '')
    .replace(/\|$/, '')
    .split('|')
    .map((c) => c.trim());
}

function looksLikeTableRow(line) {
  return line.includes('|') && (line.match(/\|/g) || []).length >= 2;
}

export function parseAnswer(text = '') {
  const lines = String(text).replace(/\r/g, '').split('\n');
  const blocks = [];
  let sec = null; // 当前所属段落：conclusion / basis / action
  let list = null;

  const flushList = () => {
    if (list && list.items.length) blocks.push(list);
    list = null;
  };

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i].trim();
    if (!line) {
      flushList();
      continue;
    }

    // ---------- 表格 ----------
    if (looksLikeTableRow(line)) {
      const header = splitRow(line);
      let j = i + 1;
      if (j < lines.length && isSepRow(lines[j].trim())) j += 1;
      const rows = [];
      while (j < lines.length && looksLikeTableRow(lines[j].trim())) {
        rows.push(splitRow(lines[j].trim()));
        j += 1;
      }
      if (rows.length) {
        flushList();
        blocks.push({ type: 'table', sec, header, rows });
        i = j - 1;
        continue;
      }
    }

    // ---------- 三段式小标题 ----------
    const found = SECTION_TITLES.find((s) => s.re.test(line));
    if (found) {
      flushList();
      sec = found.kind;
      blocks.push({ type: 'heading', kind: found.kind, text: found.label });
      const rest = line.replace(found.re, '').trim();
      if (rest) blocks.push({ type: 'paragraph', sec, inline: tokenizeInline(rest) });
      continue;
    }

    // ---------- 依据：逐条引文卡 ----------
    if (sec === 'basis') {
      const basis = parseBasisLine(line);
      if (basis) {
        flushList();
        // 引文正文末尾可能还有多余冒号/引号，交给样式处理
        blocks.push({
          type: 'basisItem',
          doc: basis.doc,
          clause: basis.clause,
          inline: tokenizeInline(basis.text),
        });
        continue;
      }
    }

    // ---------- 列表 ----------
    const ordered = ORDERED_RE.test(line);
    if (ordered || BULLET_RE.test(line)) {
      const body = line.replace(ordered ? ORDERED_RE : BULLET_RE, '');
      if (!list || list.ordered !== ordered) {
        flushList();
        list = { type: 'list', sec, ordered, items: [] };
      }
      list.items.push(tokenizeInline(body));
      continue;
    }

    // ---------- 普通段落 ----------
    flushList();
    blocks.push({ type: 'paragraph', sec, inline: tokenizeInline(line) });
  }

  flushList();
  return blocks;
}

/** 生成纯文本版本（复制用）：去掉 Markdown 标记，表格用制表符对齐。 */
export function toPlainText(text = '') {
  return parseAnswer(text)
    .map((b) => {
      if (b.type === 'heading') return `【${b.text}】`;
      if (b.type === 'basisItem') {
        const head = [b.doc, b.clause].filter(Boolean).join(' ');
        return `- ${head}${head ? '：' : ''}${b.inline.map((x) => x.v).join('')}`;
      }
      if (b.type === 'paragraph') return b.inline.map((x) => x.v).join('');
      if (b.type === 'list')
        return b.items
          .map((it, k) => `${b.ordered ? `${k + 1}.` : '·'} ${it.map((x) => x.v).join('')}`)
          .join('\n');
      if (b.type === 'table')
        return [b.header.join('\t'), ...b.rows.map((r) => r.join('\t'))].join('\n');
      return '';
    })
    .join('\n');
}
