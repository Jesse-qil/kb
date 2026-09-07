/* ==========================================================================
   管理页（manage.html）逻辑
   依赖：common.js（esc / toast / toggleSidebar）
   分区：数据加载 → 渲染 → 多选/导出 → 预览 → 标签管理 → 工具 → 启动
   ========================================================================== */

const selectedDocs = new Set();
let allTags = [];

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
    loadTags();
  } catch (e) {
    document.getElementById("topicDist").innerHTML = '<div class="empty-tip">⚠️ 统计加载失败：' + esc(e.message) + "</div>";
    toast("统计加载失败：" + e.message, "error");
  }
}

async function searchDocs(q) {
  const tbody = document.getElementById("docTbody");
  tbody.innerHTML = '<tr><td colspan="6" class="loading">搜索中…</td></tr>';
  try {
    const r = await fetch("/api/docs?q=" + encodeURIComponent(q));
    const d = await r.json();
    renderDocs(d.docs || []);
  } catch (e) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-tip">⚠️ 搜索失败：' + esc(e.message) + "</td></tr>";
  }
}

async function triggerIngest() {
  try {
    openImportModal("📥 一键入库");
    const r = await fetch("/api/ingest", { method: "POST" });
    await r.json();
    pollIngestProgress(() => { loadAll(); toast("入库完成，已刷新台账", "success"); });
  } catch (e) {
    closeImportModal();
    toast("入库触发失败：" + e.message, "error");
  }
}

/* ===== 导入进度弹窗 ===== */
let importBusy = false;   // 上传或入库进行中（禁止关闭弹窗）
let ingestFailCount = 0;  // 入库轮询连续失败计数
function el(id) { return document.getElementById(id); }

function updateForceCloseBtn() {
  const btn = el("forceCloseBtn");
  if (btn) btn.style.display = importBusy ? "inline-block" : "none";
}

function forceCloseImport() {
  if (!confirm("确定要强制关闭吗？\n\n入库可能仍在后台运行，关闭后可在管理页手动刷新查看结果。")) return;
  importBusy = false;
  if (ingestTimer) { clearInterval(ingestTimer); ingestTimer = null; }
  updateForceCloseBtn();
  const m = el("importModal");
  if (m) m.style.display = "none";
  toast("已强制关闭弹窗，入库仍在后台运行，可手动刷新查看结果", "warning");
}

function openImportModal(title) {
  const m = el("importModal");
  if (m) m.style.display = "flex";
  const t = el("importModalTitle");
  if (t && title) t.textContent = title;
}

function closeImportModal(e) {
  if (e && e.target.id !== "importModal") return;
  if (importBusy) { toast("任务进行中，请稍候…", "error"); return; }
  const m = el("importModal");
  if (m) m.style.display = "none";
}

/* ===== 导入（文件/文件夹选择 → 上传 → 一键入库） ===== */
function pickImportFiles() {
  const inp = el("importFileInput");
  if (!inp) return;
  inp.value = "";
  inp.click();
}

function pickImportFolder() {
  const inp = el("importFolderInput");
  if (!inp) return;
  inp.value = "";
  inp.click();
}

function handleImportFiles(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return;
  uploadFiles(files);
}

function handleImportFolder(input) {
  const files = Array.from(input.files || []);
  if (!files.length) return;
  // 文件夹可能很大：超过 300 个文件时按子集提示（避免单次请求过大）
  if (files.length > 300) {
    toast("本次选择 " + files.length + " 个文件，超过单批上限，将取前 300 个", "error");
    files.length = 300;
  }
  uploadFiles(files);
}

function setUploadProgress(pct, stage) {
  const row = el("uploadProgRow"), bar = el("uploadProgBar"),
        pctEl = el("uploadProgPct"), stageEl = el("uploadProgStage");
  if (row) row.style.display = "block";
  if (bar) bar.style.width = Math.max(0, Math.min(100, pct)) + "%";
  if (pctEl) pctEl.textContent = Math.round(pct) + "%";
  if (stageEl) stageEl.textContent = stage || "上传中…";
}

function uploadFiles(files) {
  if (!files.length) return;
  const btn = el("importPickBtn");
  if (btn) btn.disabled = true;
  openImportModal("📤 正在导入 " + files.length + " 个文件…");
  importBusy = true;
  updateForceCloseBtn();
  const resultsEl = el("importResults");
  if (resultsEl) resultsEl.innerHTML = "";
  setUploadProgress(0, "正在上传 " + files.length + " 个文件…");

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  fd.append("auto", "1");          // 一键入库：上传即自动分类入库

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/upload");
  xhr.upload.onprogress = e => {
    if (e.lengthComputable) {
      setUploadProgress(e.loaded / e.total * 100, "正在上传 " + files.length + " 个文件…");
    }
  };
  xhr.onload = () => {
    if (btn) btn.disabled = false;
    importBusy = false;
    updateForceCloseBtn();
    let data;
    try { data = JSON.parse(xhr.responseText); } catch (e) { data = null; }
    if (xhr.status !== 200 || !data) {
      setUploadProgress(100, "上传失败");
      const pctEl = el("uploadProgPct");
      if (pctEl) pctEl.textContent = "失败";
      toast("上传失败：" + ((data && data.detail) || xhr.statusText || "网络错误"), "error");
      return;
    }
    const results = data.results || [];
    renderImportResults(results);
    const done = results.filter(r => r.status === "done").length;
    const dup = results.filter(r => r.status === "duplicate").length;
    const pend = results.filter(r => r.status === "pending").length;
    const err = results.filter(r => r.status === "error").length;
    if (data.ingest_started) {
      setUploadProgress(100, "上传完成，等待入库…");
      importBusy = true;
      updateForceCloseBtn();
      ingestFailCount = 0;
      // 入库超时保护：10分钟后强制重置 importBusy，避免弹窗永远关不掉
      const ingestTimeout = setTimeout(() => {
        if (importBusy) {
          importBusy = false;
          updateForceCloseBtn();
          toast("入库超时（10分钟），已强制解锁弹窗。可手动刷新查看结果。", "error");
        }
      }, 600000);
      pollIngestProgress(() => {
        clearTimeout(ingestTimeout);
        importBusy = false;
        updateForceCloseBtn();
        loadAll();
        toast("入库完成，已刷新台账", "success");
      });
    } else {
      setUploadProgress(100, "上传完成");
      importBusy = false;
      updateForceCloseBtn();
      setTimeout(() => {
        const row = el("uploadProgRow");
        if (row) row.style.display = "none";
      }, 2500);
    }
    const parts = [];
    if (done) parts.push("入库 " + done + " 个");
    if (dup) parts.push("重复跳过 " + dup + " 个");
    if (pend) parts.push("待审查 " + pend + " 个");
    if (err) parts.push("失败 " + err + " 个");
    toast((parts.length ? parts.join("，") : "没有可入库文件") + (err ? "（详见列表）" : ""), err ? "error" : "success");
  };
  xhr.onerror = () => {
    if (btn) btn.disabled = false;
    importBusy = false;
    updateForceCloseBtn();
    setUploadProgress(100, "网络错误");
    const pctEl = el("uploadProgPct");
    if (pctEl) pctEl.textContent = "失败";
    toast("上传失败：网络错误", "error");
  };
  xhr.send(fd);
}

function renderImportResults(results) {
  const box = el("importResults");
  if (!box) return;
  if (!results.length) { box.innerHTML = ""; return; }
  const icon = s => s === "done" ? "✅" : s === "duplicate" ? "🔁" : s === "pending" ? "📋" : "❌";
  const label = s => s === "done" ? "已入库" : s === "duplicate" ? "重复跳过" : s === "pending" ? "待审查" : "失败";
  box.innerHTML = results.map(r =>
    '<div class="import-result-item ' + (r.status === "error" ? "err" : "") + '">' +
      "<span class=\"import-result-icon\">" + icon(r.status) + "</span>" +
      '<span class="import-result-name" title="' + esc(r.filename || "") + '">' + esc(r.filename || "") + "</span>" +
      '<span class="import-result-status">' + label(r.status) +
        (r.status === "done" && r.topic ? " → [" + esc(r.topic) + "]" : "") +
        (r.status === "duplicate" && r.error ? "（" + esc(r.error) + "）" : "") +
        (r.status === "error" && r.error ? "：" + esc(r.error) : "") +
      "</span>" +
    "</div>"
  ).join("");
  box.scrollTop = box.scrollHeight;
}

/* ===== 入库进度轮询 ===== */
let ingestTimer = null;
function pollIngestProgress(onDone) {
  openImportModal();
  const row = el("ingestProgRow"), bar = el("ingestProgBar"),
        pctEl = el("ingestProgPct"), stageEl = el("ingestProgStage"), metaEl = el("ingestProgMeta");
  if (row) row.style.display = "block";
  if (bar) bar.style.width = "0%";
  if (pctEl) pctEl.textContent = "0%";
  if (stageEl) stageEl.textContent = "排队中…";
  if (metaEl) metaEl.textContent = "";
  if (ingestTimer) clearInterval(ingestTimer);
  ingestTimer = setInterval(async () => {
    try {
      const r = await fetch("/api/ingest/progress");
      const d = await r.json();
      const pct = d.percent ?? 0;
      if (bar) bar.style.width = pct + "%";
      if (pctEl) pctEl.textContent = pct + "%";
      if (stageEl) stageEl.textContent = d.stage || "入库中…";
      const meta = [];
      if (d.done && d.total) meta.push(d.done + "/" + d.total + " 个");
      if (d.current) meta.push(d.current);
      if (d.elapsed) meta.push("已用 " + d.elapsed + "s");
      if (pct < 100 && d.eta) meta.push("预计剩余 " + d.eta + "s");
      if (metaEl) metaEl.textContent = meta.join(" · ");
      if (!d.running) {
        clearInterval(ingestTimer); ingestTimer = null;
        if (d.error) {
          if (stageEl) stageEl.textContent = "❌ 出错：" + d.error;
          toast("入库出错：" + d.error, "error");
        } else {
          if (stageEl) stageEl.textContent = "✅ 入库完成";
          if (pctEl) pctEl.textContent = "100%";
          const s = d.result || {};
          if (s && metaEl) {
            const parts = [];
            if (s.new) parts.push("新增 " + s.new);
            if (s.changed) parts.push("修改 " + s.changed);
            if (s.removed) parts.push("删除 " + s.removed);
            if (s.unchanged) parts.push("未变 " + s.unchanged);
            metaEl.textContent =
              (parts.length ? parts.join("，") : "无变化") + (s.chunks ? "，共 " + s.chunks + " 片段" : "");
          }
        }
        setTimeout(() => {
          if (!ingestTimer && row) row.style.display = "none";
        }, 4000);
        if (onDone) onDone();
      }
    } catch (e) {
      console.error("轮询入库进度失败:", e);
      // 连续失败3次后强制解锁，避免弹窗永远关不掉
      ingestFailCount = (ingestFailCount || 0) + 1;
      if (ingestFailCount >= 3) {
        clearInterval(ingestTimer);
        ingestTimer = null;
        importBusy = false;
        updateForceCloseBtn();
        toast("入库进度查询连续失败，已强制解锁弹窗。可手动刷新查看结果。", "error");
        if (onDone) onDone();
      }
    }
  }, 600);
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
  if (!docs.length) { tbody.innerHTML = '<tr><td colspan="6" class="empty-tip">没有匹配的文档</td></tr>'; return; }
  tbody.innerHTML = docs.map(d => {
    const checked = selectedDocs.has(d.path) ? "checked" : "";
    return "<tr>" +
      '<td class="doc-check"><input type="checkbox" ' + checked + ' onchange="toggleSelectDoc(\'' + esc(d.path).replace(/'/g, "\\'") + '\', this.checked)"></td>' +
      '<td><div class="doc-filename" title="' + esc(d.path) + '" onclick="previewDoc(\'' + esc(d.path).replace(/'/g, "\\'") + '\', \'' + esc(d.filename).replace(/'/g, "\\'") + '\')">' + esc(d.filename) + "</div></td>" +
      '<td><span class="doc-topic">' + esc(d.topic) + "</span></td>" +
      "<td>" + (d.tags || []).map(t => '<span class="doc-tag">' + esc(t) + "</span>").join("") + "</td>" +
      '<td class="doc-time">' + fmtTime(d.mtime) + "</td>" +
      '<td class="doc-actions">' +
        '<button class="doc-view-btn" onclick="previewDoc(\'' + esc(d.path).replace(/'/g, "\\'") + '\', \'' + esc(d.filename).replace(/'/g, "\\'") + '\')" title="预览">👁</button>' +
        '<button class="doc-export-btn" onclick="exportDoc(\'' + esc(d.path).replace(/'/g, "\\'") + '\', \'' + esc(d.filename).replace(/'/g, "\\'") + '\')" title="导出此文档">⬇</button>' +
        '<button class="doc-del-btn" onclick="deleteDoc(\'' + esc(d.path).replace(/'/g, "\\'") + '\', \'' + esc(d.filename).replace(/'/g, "\\'") + '\')" title="从知识库移除">🗑</button>' +
      "</td>" +
    "</tr>";
  }).join("");
}

/* ===== 多选与批量导出 ===== */
function toggleSelectDoc(path, checked) {
  if (checked) selectedDocs.add(path); else selectedDocs.delete(path);
  updateSelectCount();
}

function toggleSelectAll(cb) {
  const rows = document.querySelectorAll("#docTbody input[type=checkbox]");
  rows.forEach(r => {
    r.checked = cb.checked;
    const path = r.closest("tr").querySelector(".doc-filename").getAttribute("title");
    if (cb.checked) selectedDocs.add(path); else selectedDocs.delete(path);
  });
  updateSelectCount();
}

function updateSelectCount() {
  document.getElementById("selectCount").textContent = "已选 " + selectedDocs.size + " 篇";
  document.getElementById("exportSelectedBtn").disabled = selectedDocs.size === 0;
}

function exportSelected() {
  if (selectedDocs.size === 0) { toast("请先勾选文档", "error"); return; }
  toast("正在打包 " + selectedDocs.size + " 篇文档…", "success");
  fetch("/api/docs/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths: Array.from(selectedDocs) })
  }).then(r => {
    if (r.ok) return r.blob();
    return r.json().then(d => { throw new Error(d.error || "导出失败"); });
  }).then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "kb_docs_export.zip";
    a.click();
    URL.revokeObjectURL(url);
    toast("导出成功", "success");
  }).catch(e => toast("导出失败：" + e.message, "error"));
}

/* ===== 单篇文档导出 ===== */
function exportDoc(path, filename) {
  toast("正在导出 " + filename + " …", "success");
  fetch("/api/docs/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ paths: [path] })
  }).then(r => {
    if (r.ok) return r.blob();
    return r.json().then(d => { throw new Error(d.error || "导出失败"); });
  }).then(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "kb_doc_export.zip";
    a.click();
    URL.revokeObjectURL(url);
    toast("导出成功", "success");
  }).catch(e => toast("导出失败：" + e.message, "error"));
}

/* ===== 文档预览 ===== */
async function previewDoc(path, filename) {
  const modal = document.getElementById("previewModal");
  const body = document.getElementById("previewBody");
  document.getElementById("previewTitle").textContent = "📄 " + filename;
  body.innerHTML = '<div class="loading">加载中…</div>';
  modal.style.display = "flex";
  try {
    const r = await fetch("/api/docs/" + encodeURIComponent(path) + "/content");
    const d = await r.json();
    if (d.ok) {
      body.innerHTML = '<pre class="preview-content">' + esc(d.content) + "</pre>";
    } else {
      body.innerHTML = '<div class="empty-tip">⚠️ ' + esc(d.error || "加载失败") + "</div>";
    }
  } catch (e) {
    body.innerHTML = '<div class="empty-tip">⚠️ 加载失败：' + esc(e.message) + "</div>";
  }
}

function closePreview(e) {
  if (e && e.target.id !== "previewModal") return;
  document.getElementById("previewModal").style.display = "none";
}

/* ===== 标签管理 ===== */
async function loadTags() {
  try {
    const r = await fetch("/api/tags");
    const d = await r.json();
    allTags = d.tags || [];
    document.getElementById("tagTotal").textContent = allTags.length;
    renderTagManageList(allTags);
    const dl = document.getElementById("tagDatalist");
    dl.innerHTML = allTags.map(t => '<option value="' + esc(t) + '">').join("");
  } catch (e) {
    document.getElementById("tagManageList").innerHTML = '<div class="empty-tip">标签加载失败</div>';
  }
}

function renderTagManageList(tags) {
  const el = document.getElementById("tagManageList");
  if (!tags.length) { el.innerHTML = '<div class="empty-tip">没有匹配的标签</div>'; return; }
  el.innerHTML = tags.map(t =>
    '<div class="tag-manage-item">' +
      '<span class="tag-manage-name">' + esc(t) + '</span>' +
      '<button class="tag-del-btn" onclick="deleteTag(\'' + esc(t).replace(/'/g, "\\'") + '\')" title="删除标签">🗑</button>' +
    "</div>"
  ).join("");
}

function filterTags(q) {
  q = q.trim().toLowerCase();
  if (!q) { renderTagManageList(allTags); return; }
  renderTagManageList(allTags.filter(t => t.toLowerCase().includes(q)));
}

async function deleteTag(tag) {
  if (!confirm("确认删除标签「" + tag + "」？\n\n注意：这只会从标签库移除该标签，文档片段上已有的标签不受影响。")) return;
  try {
    const r = await fetch("/api/tags/" + encodeURIComponent(tag), { method: "DELETE" });
    const d = await r.json();
    if (d.ok) {
      toast("标签已删除：" + tag, "success");
      loadTags();
      loadStats();
    } else {
      toast("删除失败：" + (d.error || "未知"), "error");
    }
  } catch (e) {
    toast("删除请求失败：" + e.message, "error");
  }
}

function showMergeDialog() {
  document.getElementById("mergeFrom").value = "";
  document.getElementById("mergeTo").value = "";
  document.getElementById("mergeModal").style.display = "flex";
}

function closeMergeDialog(e) {
  if (e && e.target.id !== "mergeModal") return;
  document.getElementById("mergeModal").style.display = "none";
}

async function doMergeTag() {
  const from = document.getElementById("mergeFrom").value.trim();
  const to = document.getElementById("mergeTo").value.trim();
  if (!from || !to) { toast("请填写源标签和目标标签", "error"); return; }
  if (from === to) { toast("两个标签相同，无需合并", "error"); return; }
  try {
    const r = await fetch("/api/tags/merge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ from, to })
    });
    const d = await r.json();
    if (d.ok) {
      toast("合并成功：" + from + " → " + to, "success");
      closeMergeDialog();
      loadTags();
    } else {
      toast("合并失败：" + (d.error || "未知"), "error");
    }
  } catch (e) {
    toast("合并请求失败：" + e.message, "error");
  }
}

/* ===== 删除文档 ===== */
async function deleteDoc(path, filename) {
  const delFile = confirm("从知识库移除「" + filename + "」？\n\n点击「确定」仅从向量库和台账移除（保留原始文件，下次重导会恢复）。\n点击「取消」则不删除。");
  if (!delFile) return;
  try {
    const res = await fetch("/api/docs/" + encodeURIComponent(path), { method: "DELETE" });
    const data = await res.json();
    if (data.ok) {
      toast("已移除：" + filename + "（剩余 " + data.remaining_docs + " 篇）", "success");
      selectedDocs.delete(path);
      loadStats();
      searchDocs(document.getElementById("docSearch").value.trim());
    } else {
      toast("删除失败：" + (data.error || "未知"), "error");
    }
  } catch (e) {
    toast("删除请求失败：" + e, "error");
  }
}

/* ===== 导出快照 ===== */
function exportSnapshot() {
  toast("正在生成快照，请稍候…", "success");
  window.location.href = "/api/manage/export";
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
  selectedDocs.clear();
  updateSelectCount();
  loadStats();
  searchDocs("");
  loadTags();
}

/* ===== 启动 ===== */
document.getElementById("docSearch").addEventListener("keydown", e => {
  if (e.key === "Enter") searchDocs(e.target.value.trim());
});

loadStats();
searchDocs("");
loadTags();
