/* ==========================================================================
   问答页（index.html）逻辑
   依赖：common.js（esc / toast / toggleSidebar / closeSidebar）
   分区：消息渲染 → Markdown → 流式问答 → 上传/导入 → 待审查 → 来源
        → 入库进度 → 会话管理 → 输入框 → 启动
   ========================================================================== */

const chatEl = document.getElementById("chat");
const sendBtn = document.getElementById("sendBtn");
const inputEl = document.getElementById("question");

/* ===== 消息渲染 ===== */
function makeRow(cls, avatar) {
  const row = document.createElement("div");
  row.className = "row " + cls;
  const av = document.createElement("div");
  av.className = "avatar " + (cls === "user" ? "user-av" : "bot-av");
  av.textContent = avatar;
  const bubble = document.createElement("div");
  bubble.className = "bubble " + (cls === "user" ? "user" : "bot");
  row.appendChild(av);
  row.appendChild(bubble);
  return { row, bubble };
}

function addMsg(text, cls) {
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
  const { row, bubble } = makeRow(cls, cls === "user" ? "🧑" : "🤖");
  bubble.innerHTML = renderMarkdown(text);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return bubble;
}

/* ===== 轻量 Markdown 渲染 ===== */
function renderMarkdown(text) {
  const blocks = [];
  text = String(text).replace(/```(\w*)\n?([\s\S]*?)```/g, (m, lang, code) => {
    blocks.push('<pre class="code"><code>' + esc(code.trim()) + '</code></pre>');
    return "\u0000" + (blocks.length - 1) + "\u0000";
  });
  let html = esc(text);
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, '<code class="inline">$1</code>');
  html = html.replace(/^### (.+)$/gm, '<div class="md-h3">$1</div>');
  html = html.replace(/^## (.+)$/gm, '<div class="md-h2">$1</div>');
  html = html.replace(/^# (.+)$/gm, '<div class="md-h1">$1</div>');
  html = html.replace(/^- (.+)$/gm, '<div class="md-li">• $1</div>');
  // 表格：| 分隔的连续行 → <table>
  html = html.replace(/(^\|.*\|$)(?:\r?\n\|[:\-\s|]+\|)?(?:\r?\n\|.*\|$)+/gm, function (m) {
    const lines = m.split(/\r?\n/).filter(function (l) { return l.trim(); });
    if (lines.length < 2) return m;
    const cells = function (line) {
      return line.split("|").slice(1, -1).map(function (c) { return c.trim(); });
    };
    const head = cells(lines[0]);
    let out = '<table class="md-table"><thead><tr>' +
      head.map(function (c) { return "<th>" + c + "</th>"; }).join("") + "</tr></thead><tbody>";
    for (let i = 2; i < lines.length; i++) {
      out += "<tr>" + cells(lines[i]).map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
    }
    return out + "</tbody></table>";
  });
  html = html.replace(/\n/g, "<br>");
  html = html.replace(/\u0000(\d+)\u0000/g, (m, i) => blocks[+i]);
  return html;
}

/* ===== 代码块复制按钮 ===== */
function attachCopyButtons() {
  document.querySelectorAll(".code:not(.has-copy)").forEach(pre => {
    pre.classList.add("has-copy");
    const btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "复制";
    btn.onclick = () => {
      const code = pre.querySelector("code");
      if (!code) return;
      navigator.clipboard.writeText(code.textContent).then(() => {
        btn.textContent = "已复制";
        setTimeout(() => btn.textContent = "复制", 1500);
      }).catch(() => toast("复制失败", "error"));
    };
    pre.appendChild(btn);
  });
}

/* ===== 流式机器人气泡 ===== */
function createBotBubble() {
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
  const { row, bubble } = makeRow("bot", "🤖");
  const chips = document.createElement("div");
  chips.className = "status-chips";
  const text = document.createElement("div");
  text.className = "bubble-text";
  const cursor = document.createElement("span");
  cursor.className = "cursor";
  bubble.appendChild(chips);
  bubble.appendChild(text);
  bubble.appendChild(cursor);
  chatEl.appendChild(row);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { row, text, cursor, chips, deltaCount: 0 };
}
function addStatusChip(b, label) {
  const c = document.createElement("span");
  c.className = "status-chip";
  c.textContent = label;
  b.chips.appendChild(c);
  chatEl.scrollTop = chatEl.scrollHeight;
}
function clearStatusChips(b) { b.chips.innerHTML = ""; }
function hideCursor(b) { b.cursor.style.display = "none"; }
function statusIcon(stage) {
  if (stage.indexOf("规划") >= 0) return "🧭";
  if (stage.indexOf("检索") >= 0) return "🔍";
  if (stage.indexOf("工具") >= 0) return "🛠";
  if (stage.indexOf("生成") >= 0) return "✍️";
  if (stage.indexOf("审查") >= 0) return "🛡";
  if (stage.indexOf("目录") >= 0) return "📖";
  if (stage.indexOf("重写") >= 0) return "✏️";
  return "⚙️";
}
function handleStreamEvent(ev, b) {
  if (ev.type === "status") {
    const s = ev.stage || "";
    if (s.indexOf("重写") >= 0) {
      b.deltaCount = 0;
      b.text.textContent = "";
      clearStatusChips(b);
      addStatusChip(b, "✏️ " + s);
    } else {
      addStatusChip(b, statusIcon(s) + " " + s);
    }
  } else if (ev.type === "delta") {
    if (b.deltaCount === 0) clearStatusChips(b);
    b.deltaCount++;
    b.text.textContent += ev.content || "";
    chatEl.scrollTop = chatEl.scrollHeight;
  } else if (ev.type === "done") {
    hideCursor(b);
    clearStatusChips(b);
    const finalText = ev.answer || b.text.textContent;
    b.text.innerHTML = renderMarkdown(finalText);
    attachCopyButtons();
    chatEl.scrollTop = chatEl.scrollHeight;
    if (ev.sources && ev.sources.length) renderSources(ev.sources);
  } else if (ev.type === "error") {
    hideCursor(b);
    b.text.textContent += "\n\n⚠️ " + (ev.content || "未知错误");
  }
}

/* ===== 流式发送 ===== */
async function send() {
  const q = inputEl.value.trim();
  if (!q) return;
  inputEl.value = "";
  inputEl.style.height = "auto";
  sendBtn.disabled = true;
  addMsg(q, "user");
  const b = createBotBubble();
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, session_id: getSessionId() }),
    });
    if (!res.ok || !res.body) throw new Error("HTTP " + res.status);
    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        if (!chunk.startsWith("data: ")) continue;
        let ev;
        try { ev = JSON.parse(chunk.slice(6)); } catch (e) { continue; }
        handleStreamEvent(ev, b);
      }
    }
  } catch (err) {
    hideCursor(b);
    clearStatusChips(b);
    b.text.innerHTML = '<span style="color:#ff8b8b">⚠️ 请求失败，请确认服务已启动</span>';
    toast("请求失败：" + err.message, "error");
  }
  sendBtn.disabled = false;
}
function sendSugg(t) { inputEl.value = t; send(); }
function sendChoice(text) { inputEl.value = text; send(); }

/* ===== 导出当前会话为 Markdown ===== */
function exportSession() {
  const rows = document.querySelectorAll("#chat .row");
  if (!rows.length) { toast("没有可导出的消息", "error"); return; }
  let md = "# 个人知识库 · 会话导出\n\n";
  md += "> 会话：" + (document.getElementById("curSessName").textContent || "未命名") + "\n";
  md += "> 导出时间：" + new Date().toLocaleString() + "\n\n";
  md += "---\n\n";
  rows.forEach(row => {
    const isUser = row.classList.contains("user");
    const bubble = row.querySelector(".bubble");
    if (!bubble) return;
    const text = bubble.textContent.trim();
    if (!text) return;
    if (isUser) {
      md += "## 🧑 用户\n\n" + text + "\n\n";
    } else {
      md += "## 🤖 AI 回答\n\n" + text + "\n\n";
    }
  });
  const srcCards = document.querySelectorAll("#chat .src-card");
  if (srcCards.length) {
    md += "---\n\n## 📎 参考来源\n\n";
    srcCards.forEach(card => {
      const files = card.querySelectorAll(".src-file-head");
      files.forEach(f => {
        const t = f.textContent.trim().replace(/\s+/g, " ");
        md += "- " + t + "\n";
      });
    });
    md += "\n";
  }
  const blob = new Blob([md], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const sessName = (document.getElementById("curSessName").textContent || "会话").replace(/[\\/:*?"<>|]/g, "_");
  a.href = url;
  a.download = sessName + "_" + new Date().toISOString().slice(0, 10) + ".md";
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  toast("已导出会话为 Markdown", "success");
}

/* ===== 上传 ===== */
async function upload() {
  const input = document.getElementById("fileInput");
  if (!input.files.length) return;
  const files = Array.from(input.files);
  const auto = document.getElementById("autoMode").checked;
  const names = files.map(f => f.name).join("、");
  addMsg((auto ? "🤖" : "📤") + " 正在上传" + (auto ? "（自动入库）" : "（待审查）") + "：" + names, "user");
  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  fd.append("auto", auto ? "1" : "0");
  let doneNames = [], pendingNames = [], dupMsg = [], errMsg = "";
  try {
    const res = await fetch("/api/upload", { method: "POST", body: fd });
    const data = await res.json();
    (data.results || []).forEach(r => {
      if (r.status === "done") doneNames.push(r.filename + " → [" + (r.topic || "默认") + "]");
      else if (r.status === "pending") pendingNames.push(r.filename + (r.suggest_topic ? " → 建议[" + r.suggest_topic + "]" : " → 主题待定"));
      else if (r.status === "duplicate") dupMsg.push(r.filename + "（" + (r.error || "重复") + "）");
      else errMsg.push(r.filename + "：" + (r.error || "失败"));
    });
    if (data.ingest_started) pollIngest();
  } catch (e) { errMsg.push("请求失败：" + e); }
  if (doneNames.length) addMsg("🤖 自动入库中：" + doneNames.join("、"), "bot");
  if (pendingNames.length) { addMsg("📋 已进入待审查：" + pendingNames.join("、") + "。点击左侧「待审查」确认入库。", "bot"); toast("有待审查文件待确认", "success"); }
  if (dupMsg.length) addMsg("🔁 已跳过重复：" + dupMsg.join("、"), "bot");
  if (errMsg.length) { addMsg("⚠️ 失败：" + errMsg.join("；"), "bot"); toast("上传失败", "error"); }
  input.value = "";
  refreshPending();
}

/* ===== 文件夹导入 ===== */
async function importFolder() {
  const path = prompt("输入要导入的本地文件夹路径（自动识别 .md/.docx/.pdf）：");
  if (!path || !path.trim()) return;
  const auto = document.getElementById("autoMode").checked;
  addMsg("📁 正在导入文件夹（" + (auto ? "自动入库" : "待审查") + "）：" + path.trim(), "user");
  let doneNames = [], pendingNames = [], dupMsg = [], errMsg = [];
  try {
    const res = await fetch("/api/import/folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path.trim(), auto: auto })
    });
    const data = await res.json();
    if (!res.ok) { errMsg.push(data.detail || "导入失败"); }
    else {
      const s = data.summary || {};
      (s.details || []).forEach(d => {
        if (d.status === "done") doneNames.push(d.file + " → [" + (d.topic || "默认") + "]");
        else if (d.status === "pending") pendingNames.push(d.file + (d.suggest_topic ? " → 建议[" + d.suggest_topic + "]" : " → 主题待定"));
        else if (d.status === "duplicate") dupMsg.push(d.file + "（" + (d.note || "重复") + "）");
        else if (d.status === "error") errMsg.push(d.file + "：" + (d.note || "失败"));
      });
      if (s.skipped) addMsg("⚠️ 超过单批上限跳过 " + s.skipped + " 个", "bot");
      if (data.ingest_started) pollIngest();
    }
  } catch (e) { errMsg.push("请求失败：" + e); }
  if (doneNames.length) addMsg("🤖 自动入库中：" + doneNames.join("、"), "bot");
  if (pendingNames.length) { addMsg("📋 已进入待审查：" + pendingNames.join("、") + "。点击左侧「待审查」确认入库。", "bot"); toast("有待审查文件待确认", "success"); }
  if (dupMsg.length) addMsg("🔁 已跳过重复：" + dupMsg.join("、"), "bot");
  if (errMsg.length) { addMsg("⚠️ 失败：" + errMsg.join("；"), "bot"); toast("导入失败", "error"); }
  refreshPending();
}

/* ===== 待审查 ===== */
function togglePending() {
  const p = document.getElementById("pendingPanel");
  p.style.display = p.style.display === "none" ? "block" : "none";
  if (p.style.display !== "none") refreshPending();
}
async function refreshPending() {
  try {
    const res = await fetch("/api/pending");
    const data = await res.json();
    const list = data.pending || [];
    const btn = document.getElementById("pendingBtn");
    const countEl = document.getElementById("pendingCount");
    const box = document.getElementById("pendingList");
    if (list.length) { btn.style.display = ""; countEl.textContent = list.length; }
    else { btn.style.display = "none"; }
    box.innerHTML = list.length ? "" : "<div style='color:#6b7a90;font-size:13px'>没有待审查文件</div>";
    list.forEach(item => {
      const row = document.createElement("div");
      row.className = "pending-row";
      const tagLine = (item.tags && item.tags.length)
        ? "<div style='flex-basis:100%;font-size:11px;color:#7d8ba1;padding-left:4px'>🏷 " + item.tags.map(t => "<span class='tag-chip'>" + esc(t) + "</span>").join("") + "</div>" : "";
      row.innerHTML =
        "<span class='fname'>📄 " + esc(item.filename) + "</span>" +
        "<input id='topic_" + esc(item.filename) + "' value='" + esc(item.suggest_topic || "") + "' placeholder='主题（可改）'>" +
        "<button onclick='reviewApprove(this)' data-file='" + esc(item.filename) + "' style='background:#2f6fed;color:#fff;border:none;padding:5px 11px;border-radius:7px;cursor:pointer;font-size:12px'>✓ 确认入库</button>" +
        "<button onclick='reviewReject(this)' data-file='" + esc(item.filename) + "' style='background:rgba(255,255,255,0.07);color:#ff8b8b;border:1px solid rgba(255,255,255,0.1);padding:5px 11px;border-radius:7px;cursor:pointer;font-size:12px'>✕ 丢弃</button>" +
        tagLine;
      box.appendChild(row);
    });
  } catch (e) {}
}
async function reviewApprove(btn) {
  const file = btn.getAttribute("data-file");
  const topicInput = document.getElementById("topic_" + file);
  const topic = topicInput.value.trim();
  if (!topic) { toast("请填写主题目录名", "error"); return; }
  btn.disabled = true; btn.textContent = "入库中…";
  try {
    const res = await fetch("/api/review", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file, topic: topic }),
    });
    const data = await res.json();
    if (res.ok) { addMsg("入库中：" + file + " → [" + topic + "]", "bot"); pollIngest(); toast("已提交入库", "success"); }
    else { toast("失败：" + (data.detail || "未知"), "error"); }
  } catch (e) { toast("请求失败：" + e, "error"); }
  refreshPending();
}
async function reviewReject(btn) {
  const file = btn.getAttribute("data-file");
  if (!confirm("丢弃 " + file + "？")) return;
  try {
    await fetch("/api/pending", {
      method: "DELETE", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: file }),
    });
    toast("已丢弃", "success");
  } catch (e) {}
  refreshPending(); stats();
}

/* ===== 参考来源 ===== */
function renderSources(sources) {
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
  const card = document.createElement("div");
  card.className = "src-card";
  const groups = {};
  sources.forEach(s => { (groups[s.source] = groups[s.source] || []).push(s); });
  const files = Object.keys(groups);
  const head = document.createElement("div");
  head.className = "src-head";
  head.innerHTML = "📎 参考来源（" + files.length + " 篇）<span class='src-arrow' style='color:#6b7a90'>▶</span>";
  head.onclick = () => {
    const body = card.querySelector(".src-body");
    const arrow = head.querySelector(".src-arrow");
    const open = body.style.display !== "none";
    body.style.display = open ? "none" : "block";
    arrow.textContent = open ? "▶" : "▼";
  };
  card.appendChild(head);
  const body = document.createElement("div");
  body.className = "src-body";
  body.style.display = "none";
  files.forEach(fname => {
    const items = groups[fname];
    const fileDiv = document.createElement("div");
    fileDiv.className = "src-file";
    const best = Math.max(...items.map(s => s.score)).toFixed(3);
    const fileHead = document.createElement("div");
    fileHead.className = "src-file-head";
    fileHead.innerHTML = "📄 " + esc(fname) + " <span class='src-best'>" + items.length + " 段 · 最高 " + best + "</span> <span style='color:#6b7a90'>▶</span>";
    fileHead.onclick = () => {
      const list = fileDiv.querySelector(".src-list");
      const arrow = fileHead.querySelector("span:last-child");
      const open = list.style.display !== "none";
      list.style.display = open ? "none" : "block";
      arrow.textContent = open ? "▶" : "▼";
    };
    fileDiv.appendChild(fileHead);
    const list = document.createElement("div");
    list.className = "src-list";
    list.style.display = "none";
    items.forEach(s => {
      const line = document.createElement("div");
      const h = (s.heading || "").replace(/^#+\s*/, "").slice(0, 30);
      line.textContent = "· " + (h || "(片段)") + " — 相似度 " + s.score.toFixed(3);
      list.appendChild(line);
    });
    fileDiv.appendChild(list);
    body.appendChild(fileDiv);
  });
  card.appendChild(body);
  chatEl.appendChild(card);
  chatEl.scrollTop = chatEl.scrollHeight;
}

/* ===== 入库进度 ===== */
let progTimer = null;
function _fmtTime(sec) {
  sec = Math.max(0, Math.floor(sec || 0));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + ":" + String(s).padStart(2, "0");
}
function showProg(stage, pct, current, elapsed, eta, done, total) {
  if (typeof current === "undefined" && typeof stage === "object" && stage) {
    const d = stage;
    return showProg(d.stage, d.percent, d.current, d.elapsed, d.eta, d.done, d.total);
  }
  const w = document.getElementById("progWrap");
  w.style.display = "block";
  document.getElementById("progStage").textContent = stage || "";
  document.getElementById("progPct").textContent = (pct || 0) + "%";
  document.getElementById("progBar").style.width = (pct || 0) + "%";
  let curText = "";
  if (done && total) curText = done + "/" + total + " · ";
  curText += (current || "").trim();
  document.getElementById("progCurrent").textContent = curText || "—";
  const t = _fmtTime(elapsed);
  const etaTxt = (pct >= 100 || !eta) ? "完成" : "剩余 " + _fmtTime(eta);
  document.getElementById("progTime").textContent = "用时 " + t + " · " + etaTxt;
}
function hideProg() {
  const w = document.getElementById("progWrap");
  setTimeout(() => { w.style.display = "none"; }, 400);
}
function pollIngest(onDone) {
  showProg("排队中…", 0, "", 0, 0, 0, 0);
  if (progTimer) clearInterval(progTimer);
  progTimer = setInterval(async () => {
    try {
      const res = await fetch("/api/ingest/progress");
      const d = await res.json();
      showProg(d.stage, d.percent, d.current, d.elapsed, d.eta, d.done, d.total);
      if (!d.running) {
        clearInterval(progTimer); progTimer = null;
        if (d.error) addMsg("⚠️ 入库出错：" + d.error, "bot");
        else if (d.result) {
          const s = d.result;
          addMsg("✅ 入库完成：新增" + s.new + " 修改" + s.changed + " 未变" + s.unchanged + " 删除" + s.removed + "，片段 " + s.chunks + (s.elapsed ? "，用时 " + s.elapsed + " 秒" : ""), "bot");
          toast("入库完成", "success");
        }
        hideProg();
        if (onDone) onDone();
        stats();
      }
    } catch (e) {}
  }, 600);
}
async function ingest() {
  showProg("开始导入…", 3);
  try {
    const res = await fetch("/api/ingest", { method: "POST" });
    await res.json();
    pollIngest();
  } catch (e) { hideProg(); addMsg("⚠️ 导入请求失败：" + e, "bot"); toast("导入失败", "error"); }
}
async function stats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    document.getElementById("stats").textContent = "📦 库内 " + data.chunks + " 片段";
  } catch (e) {}
}

/* ===== 会话管理 ===== */
function getSessionId() {
  let sid = localStorage.getItem("kb_session_id");
  if (!sid) sid = "s_demo_default";
  return sid;
}
function setCurSession(sid, name) {
  localStorage.setItem("kb_session_id", sid);
  const el = document.getElementById("curSessName");
  if (el) el.textContent = name ? name.slice(0, 20) : "新会话";
}
async function newSession() {
  try {
    const res = await fetch("/api/sessions", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "新会话" }),
    });
    const data = await res.json();
    setCurSession(data.session_id, data.title);
    resetChatArea();
    addMsg("🆕 已创建新会话，之前的会话还在左侧列表里。", "bot");
    loadSessionList();
    toast("已创建新会话", "success");
  } catch (e) { toast("创建会话失败：" + e, "error"); }
}
async function switchSession(sid, title) {
  setCurSession(sid, title);
  resetChatArea();
  addMsg("已切换到会话「" + (title || "新会话").slice(0, 15) + "」，开始提问吧。", "bot");
  closeSidebar();
}
function resetChatArea() {
  document.getElementById("chat").innerHTML =
    '<div class="welcome"><div class="wl-logo">📚</div><h2>个人知识库</h2><p>问任何关于你笔记的问题，回答会流式生成</p><div class="sugg">' +
    '<button class="sugg-btn" onclick="sendSugg(\'知识库里有什么内容？\')">📖 库里有什么</button>' +
    '<button class="sugg-btn" onclick="sendSugg(\'list 和 tuple 有什么区别？\')">🐍 Python 基础</button>' +
    '<button class="sugg-btn" onclick="sendSugg(\'LangGraph 多智能体怎么做？\')">🤖 AI-Agent</button></div></div>';
}
async function loadSessionList() {
  try {
    const res = await fetch("/api/sessions");
    const data = await res.json();
    const cur = getSessionId();
    const box = document.getElementById("sessionList");
    const list = data.sessions || [];
    if (!list.length) { box.innerHTML = "<div style='color:#6b7a90;font-size:13px;padding:8px'>还没有会话</div>"; return; }
    box.innerHTML = "";
    list.forEach(s => {
      const row = document.createElement("div");
      row.className = "sess-item" + (s.id === cur ? " active" : "");
      const main = document.createElement("div");
      main.className = "sess-main";
      const t1 = document.createElement("div");
      t1.className = "sess-title";
      t1.textContent = s.title || "新会话";
      const t2 = document.createElement("div");
      t2.className = "sess-count";
      t2.textContent = s.msg_count + " 条";
      main.appendChild(t1); main.appendChild(t2);
      main.addEventListener("click", () => switchSession(s.id, s.title));
      const del = document.createElement("button");
      del.className = "sess-del";
      del.textContent = "✕"; del.title = "删除";
      del.addEventListener("click", e => { e.stopPropagation(); deleteSession(s.id); });
      row.appendChild(main); row.appendChild(del);
      box.appendChild(row);
    });
    const curS = list.find(x => x.id === cur);
    const el = document.getElementById("curSessName");
    if (el) el.textContent = curS ? (curS.title || "新会话") : "新会话";
  } catch (e) {}
}
async function deleteSession(sid) {
  if (!confirm("删除这个会话及其历史？")) return;
  try {
    await fetch("/api/sessions/" + encodeURIComponent(sid), { method: "DELETE" });
    if (getSessionId() === sid) {
      setCurSession("s_demo_default", "");
      resetChatArea();
    }
    loadSessionList();
    toast("已删除会话", "success");
  } catch (e) { toast("删除失败", "error"); }
}

/* ===== 输入框自适应高度 + Enter 发送 ===== */
inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 130) + "px";
});
inputEl.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
});

/* ===== 启动 ===== */
stats();
refreshPending();
loadSessionList();
