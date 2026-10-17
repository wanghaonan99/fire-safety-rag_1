'use client';

/**
 * 通用抽屉：会话（左）、设置（右）共用。
 *
 * 为什么用抽屉而不是再塞两个 tab：
 *   问答是主任务（90% 时间在里面），会话与设置属于"切换/调整"，不该和主任务平级；
 *   抽屉在窄屏天然可用，不需要额外的移动端导航方案。
 */

import { useEffect } from 'react';

export default function Drawer({ side = 'right', title, subtitle, onClose, children }) {
  // Esc 关闭（与 ChatTab 的 Esc 关出处面板互不冲突：抽屉打开时焦点在抽屉内）
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose?.();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <>
      <div className="drawer-mask" onClick={onClose} />
      <aside className={`drawer drawer-${side}`} role="dialog" aria-label={title}>
        <div className="drawer-head">
          <div>
            <div className="drawer-title">{title}</div>
            {subtitle ? <div className="drawer-sub">{subtitle}</div> : null}
          </div>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="关闭">
            ✕
          </button>
        </div>
        <div className="drawer-body">{children}</div>
      </aside>
    </>
  );
}
