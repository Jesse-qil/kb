/* ==========================================================================
   管理页（manage.html）逻辑
   依赖：common.js（esc / toast / toggleSidebar）
   分区：数据加载 → 渲染 → 工具 → 启动
   ========================================================================== */

/* ===== 数据加载 ===== */
async function loadStats() {
  try {
    const r = await fetch("/api/manage/stats");
    const d = await r.json();
    document.getElementById("statDocs").textContent = d.total_docs ?? 0;
    document.getElementById("statChunks").textContent = d.total_chunks ?? 0;
    document.getElementById("statTags").textContent = d.total_tags ?? 0;
    document.getElementById("statTopics").textContent = Object.keys(d.topic_distribution || {}).length;
    renderTopics(d.topic_distribution || {});
    renderTags(d.tag_top20 || []);
    renderDocs(d.docs || []);
  } catch (e) {
    document.getElementById("topicDist").innerHTML = '<div class="empty-tip">⚠️ 统计加载失败：' + esc(e.message) + "</div>";
    toast("统计加载失败：" + e.message, "error");
  }
}

async function searchDocs(q) {
  const tbody = document.getElementById("docTbody");
  tbody.innerHTML = '<tr><td colspan="4" class="loading">搜索中…</td></tr>';
  try {
    const r = await fetch("/api/docs?q=" + encodeURIComponent(q));
    const d = await r.json();
    renderDocs(d.docs || []);
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="4" class="empty-tip">⚠️ 搜索失败：' + esc(e.message) + "</td></tr>";
  }
}

async function triggerIngest() {
  try {
    const r = await fetch("/api/ingest", { method: "POST" });
    const d = await r.json();
    toast(d.started ? "已开始增量入库，稍后点刷新查看" : "正在入库中，请稍候", "success");
  } catch (e) {
    toast("入库触发失败：" + e.message, "error");
  }
}

/* ===== 渲染 ===== */
function renderTopics(dist) {
  const el = document.getElementById("topicDist");
  const entries = Object.entries(dist);
  if (!entries.length) { el.innerHTML = '<div class="empty-tip">暂无数据</div>'; return; }
  const max = Math.max(...entries.map(([, v]) => v), 1);
  el.innerHTML = entries.map(([name, cnt]) =>
    '<div class="topic-row">' +
      '<div class="topic-name" title="' + esc(name) + '">' + esc(name) + "</div>" +
      '<div class="topic-bar-wrap"><div class="topic-bar" style="width:' + Math.round(cnt / max * 100) + '%"></div></div>' +
      '<div class="topic-count">' + cnt + "</div>" +
    "</div>"
  ).join("");
}

function renderTags(tags) {
  const el = document.getElementById("tagCloud");
  if (!tags.length) { el.innerHTML = '<div class="empty-tip">暂无标签</div>'; return; }
  el.innerHTML = tags.map(t =>
    '<span class="tag-chip" onclick="filterTag(\'' + esc(t.name).replace(/'/g, "\\'") + '\')">' +
      esc(t.name) + '<span class="cnt">' + t.count + "</span></span>"
  ).join("");
}

function renderDocs(docs) {
  const tbody = document.getElementById("docTbody");
  if (!docs.length) { tbody.innerHTML = '<tr><td colspan="4" class="empty-tip">没有匹配的文档</td></tr>'; return; }
  tbody.innerHTML = docs.map(d =>
    "<tr>" +
      '<td><div class="doc-filename" title="' + esc(d.path) + '">' + esc(d.filename) + "</div></td>" +
      '<td><span class="doc-topic">' + esc(d.topic) + "</span></td>" +
      "<td>" + (d.tags || []).map(t => '<span class="doc-tag">' + esc(t) + "</span>").join("") + "</td>" +
      '<td class="doc-time">' + fmtTime(d.mtime) + "</td>" +
    "</tr>"
  ).join("");
}

/* ===== 工具 ===== */
function fmtTime(ts) {
  if (!ts) return "–";
  const d = new Date(ts * 1000);
  const p = n => String(n).padStart(2, "0");
  return d.getFullYear() + "-" + p(d.getMonth() + 1) + "-" + p(d.getDate()) + " " + p(d.getHours()) + ":" + p(d.getMinutes());
}
function filterTag(tag) {
  const box = document.getElementById("docSearch");
  box.value = tag;
  searchDocs(tag);
}
function loadAll() {
  document.getElementById("docSearch").value = "";
  loadStats();
}

/* ===== 启动 ===== */
document.getElementById("docSearch").addEventListener("keydown", e => {
  if (e.key === "Enter") searchDocs(e.target.value.trim());
});

loadStats();
