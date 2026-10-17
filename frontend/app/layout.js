export const metadata = {
  title: '工厂消防知识问答助手',
  description: '基于消防法规、应急预案与培训资料的 RAG 问答助手',
};

import './globals.css';

/**
 * 首屏防闪脚本（在 React 水合前同步执行）。
 *
 * 为什么需要：主题/字号/背景开关都存在 localStorage 里，而 React 的 useEffect
 * 要等水合之后才跑。不提前设置的话，深色用户会先看到一帧亮色（FOUC）。
 * 这里在 <head> 里同步读取偏好并写到 <html> 上，水合时状态已经一致。
 *
 * 注意：localStorage 存的是 JSON（saveLocal 用 JSON.stringify），所以要 JSON.parse。
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
