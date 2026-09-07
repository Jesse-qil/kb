/* ==========================================================================
   kb-v2 前端公共脚本
   esc / toast / toggleSidebar —— index.js 与 manage.js 共用
   依赖：页面需包含 #toast-container、#sidebar、#sidebarMask
   ========================================================================== */

/* ===== HTML 安全转义 ===== */
function esc(s) {
  return String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/* ===== Toast 提示 ===== */
function toast(msg, type) {
  const c = document.getElementById("toast-container");
  if (!c) return;
  const t = document.createElement("div");
  t.className = "toast" + (type ? " " + type : "");
  t.textContent = msg;
  c.appendChild(t);
  setTimeout(() => {
    t.style.opacity = "0";
    t.style.transition = "opacity .3s";
    setTimeout(() => t.remove(), 300);
  }, 2800);
}

/* ===== 侧边栏切换（移动端抽屉） ===== */
function toggleSidebar() {
  const sb = document.getElementById("sidebar");
  const mask = document.getElementById("sidebarMask");
  if (!sb) return;
  sb.classList.toggle("open");
  if (mask) mask.classList.toggle("show");
}
function closeSidebar() {
  const sb = document.getElementById("sidebar");
  const mask = document.getElementById("sidebarMask");
  if (sb) sb.classList.remove("open");
  if (mask) mask.classList.remove("show");
}

/* ===== 页面切换动画：点击导航先淡出再跳转（三页共用） ===== */
(function () {
  document.addEventListener("click", function (e) {
    const a = e.target && e.target.closest ? e.target.closest("a.nav-item") : null;
    if (!a) return;
    if (a.classList.contains("active")) return;   // 当前页不跳
    const href = a.getAttribute("href");
    if (!href || href === "#") return;
    e.preventDefault();
    document.body.classList.add("leaving");
    setTimeout(function () { window.location.href = href; }, 160);
  });
})();
