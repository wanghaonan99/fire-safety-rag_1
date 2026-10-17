export const metadata = {
  title: '工厂消防知识问答助手',
  description: '基于消防法规、应急预案与培训资料的 RAG 问答助手',
};

/**
 * 移动端视口配置：
 *   viewportFit=cover → 支持 iPhone 刘海屏，配合 CSS 的 env(safe-area-inset-*) 使用
 *   不限制 maximumScale → 保留双指缩放（无障碍），输入框另用 16px 字号防自动放大
 */
export const viewport = {
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#efe9e1' },
    { media: '(prefers-color-scheme: dark)', color: '#0c0e13' },
  ],
};

import './globals.css';

/**
 * 首屏防闪脚本（在 React 水合前同步执行）：主题/字号/背景开关存在 localStorage，
 * useEffect 要等水合后才跑，不提前设置会让深色用户先看到一帧亮色（FOUC）。
 */
const bootScript = `(function(){try{
  var d=document.documentElement;
  var t=localStorage.getItem('fsrag.theme');
  t=t?JSON.parse(t):'auto';
  if(!t||t==='auto'){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}
  d.dataset.theme=(t==='dark')?'dark':'light';
  var fs=localStorage.getItem('fsrag.font_scale');
  if(fs){d.style.setProperty('--fs-boost',String(JSON.parse(fs)));}
  var bg=localStorage.getItem('fsrag.bg');
  d.dataset.bg=(bg&&JSON.parse(bg)==='off')?'off':'on';
}catch(e){}})();`;

export default function RootLayout({ children }) {
  return (
    // suppressHydrationWarning：上面的脚本会在水合前修改 <html> 属性，属预期行为
    <html lang="zh-CN" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: bootScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}