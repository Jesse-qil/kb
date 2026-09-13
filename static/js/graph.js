/* ==========================================================================
   kb-v2 知识图谱页：拉取 /api/graph → Canvas 力导向图
   交互：类型筛选 / 拖拽节点 / 悬停高亮邻居 + 提示
   纯 JS 自绘（无外部库），物理模拟：斥力 + 弹簧 + 中心引力 + 阻尼
   ========================================================================== */

(function () {
  "use strict";

  var TYPE_COLOR = {
    doc: "#4DA3FF",
    topic: "#9EACEA",
    tag: "#F4B393",
    entity: "#94D8C3",
  };
  var TYPE_RADIUS = { doc: 9, topic: 13, tag: 7, entity: 6 };
  var TYPE_LABEL = { doc: "文档", topic: "领域", tag: "标签", entity: "实体" };

  var canvas, ctx, wrap, tip;
  var nodes = [], edges = [];
  var visibleTypes = { doc: true, topic: true, tag: true, entity: true };
  var layout = [];           // {x, y, vx, vy, r, color, name, type, topic, deg}
  var layoutById = {};
  var W = 0, H = 0, dpr = 1;
  var raf = null, running = false;
  var dragIdx = -1, hoverIdx = -1;
  var settled = 0;

  function $(id) { return document.getElementById(id); }

  function init() {
    canvas = $("graphCanvas");
    wrap = $("graphWrap");
    tip = $("gTip");
    ctx = canvas.getContext("2d");
    resize();
    window.addEventListener("resize", function () {
      resize();
      if (layout.length) settle(false);
    });

    // 类型筛选
    document.querySelectorAll("#ctrlRow input[data-t]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        visibleTypes[cb.getAttribute("data-t")] = cb.checked;
        applyFilter();
      });
    });

    // 拖拽 / hover
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", function () { hoverIdx = -1; });

    loadGraph();
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    W = wrap.clientWidth || 800;
    H = wrap.clientHeight || 600;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    canvas.style.width = W + "px";
    canvas.style.height = H + "px";
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ==================== 数据加载 ==================== */

  function loadGraph() {
    setStatus("加载中…");
    fetch("/api/graph")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok || data.empty) {
          nodes = []; edges = [];
          $("emptyTip").style.display = "flex";
          $("statRow").innerHTML = "";
          setStatus("");
          return;
        }
        $("emptyTip").style.display = "none";
        nodes = data.nodes || [];
        edges = data.edges || [];
        renderStats(data.meta || {});
        applyFilter();
        setStatus("节点 " + nodes.length + " · 边 " + edges.length);
      })
      .catch(function (e) {
        toast("图谱加载失败: " + e, "error");
        setStatus("");
      });
  }

  function renderStats(meta) {
    var types = {};
    nodes.forEach(function (n) { types[n.type] = (types[n.type] || 0) + 1; });
    var cards = [
      ["节点", nodes.length],
      ["边", edges.length],
      ["实体三元组", meta.triples || 0],
      ["文档", meta.docs || 0],
    ];
    var html = "";
    cards.forEach(function (c) {
      html += '<div class="stat-card"><div class="num">' + c[1] + '</div>' +
              '<div class="lbl">' + c[0] + "</div></div>";
    });
    ["doc", "topic", "tag", "entity"].forEach(function (t) {
      html += '<div class="stat-card"><div class="num" style="font-size:14px;color:' +
              TYPE_COLOR[t] + ';">' + (types[t] || 0) + "</div>" +
              '<div class="lbl">' + TYPE_LABEL[t] + "</div></div>";
    });
    $("statRow").innerHTML = html;
  }

  /* ==================== 布局构建 ==================== */

  function applyFilter() {
    var keep = {}, keepE = [];
    nodes.forEach(function (n) {
      if (visibleTypes[n.type]) keep[n.id] = true;
    });
    edges.forEach(function (e) {
      if (keep[e.src] && keep[e.dst]) keepE.push(e);
    });
    buildLayout(nodes.filter(function (n) { return keep[n.id]; }), keepE);
  }

  function buildLayout(visNodes, visEdges) {
    var n = visNodes.length;
    layout = visNodes.map(function (nd, i) {
      // 初始位置：环形分布（避免随机重叠）
      var ang = (i / Math.max(n, 1)) * Math.PI * 2;
      var rad = Math.min(W, H) * 0.32;
      return {
        id: nd.id, name: nd.name, type: nd.type, topic: nd.topic,
        r: TYPE_RADIUS[nd.type] || 6, color: TYPE_COLOR[nd.type] || "#999",
        x: W / 2 + Math.cos(ang) * rad + (Math.random() - 0.5) * 30,
        y: H / 2 + Math.sin(ang) * rad + (Math.random() - 0.5) * 30,
        vx: 0, vy: 0, deg: 0,
      };
    });
    layoutById = {};
    layout.forEach(function (nd) { layoutById[nd.id] = nd; });
    edges = visEdges;
    // 计算度（显示用）
    var deg = {};
    visEdges.forEach(function (e) {
      deg[e.src] = (deg[e.src] || 0) + 1;
      deg[e.dst] = (deg[e.dst] || 0) + 1;
    });
    layout.forEach(function (nd) { nd.deg = deg[nd.id] || 0; });
    settled = 0;
    settle(true);
  }

  /* ==================== 物理模拟与渲染 ==================== */

  function settle(animate) {
    if (!animate) {
      // 静默收敛几轮后直接画
      for (var i = 0; i < 200; i++) step();
      draw();
      return;
    }
    if (raf) cancelAnimationFrame(raf);
    running = true;
    var last = performance.now();
    function loop(now) {
      var dt = Math.min(now - last, 50) / 16.7;   // 归一化到 60fps
      last = now;
      step(dt);
      draw();
      settled++;
      if (settled < 900) {
        raf = requestAnimationFrame(loop);
      } else {
        running = false;
        raf = null;
      }
    }
    raf = requestAnimationFrame(loop);
  }

  function step(dt) {
    var n = layout.length;
    if (!n) return;
    var cx = W / 2, cy = H / 2;
    var kRep = 4200 * Math.sqrt(n) / Math.max(n, 10);   // 斥力强度随规模自适应
    var kSpr = 0.02, rest = 70;
    var kGrav = 0.012;
    var minD = 14;

    // 斥力：O(n²)
    for (var i = 0; i < n; i++) {
      var a = layout[i];
      for (var j = i + 1; j < n; j++) {
        var b = layout[j];
        var dx = a.x - b.x, dy = a.y - b.y;
        var d2 = dx * dx + dy * dy;
        if (d2 < 1) { dx = (Math.random() - 0.5) * 2; dy = (Math.random() - 0.5) * 2; d2 = 1; }
        var d = Math.sqrt(d2);
        if (d < minD) { d = minD; d2 = d * d; }
        var f = kRep / d2;
        var fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }

    // 弹簧：只对可见边
    for (var k = 0; k < edges.length; k++) {
      var e = edges[k];
      var na = layoutById[e.src], nb = layoutById[e.dst];
      if (!na || !nb) continue;
      var ex = nb.x - na.x, ey = nb.y - na.y;
      var ed = Math.sqrt(ex * ex + ey * ey) || 1;
      var f2 = kSpr * (ed - rest);
      var fx2 = (ex / ed) * f2, fy2 = (ey / ed) * f2;
      na.vx += fx2; na.vy += fy2;
      nb.vx -= fx2; nb.vy -= fy2;
    }

    // 引力 + 阻尼 + 位移
    for (var m = 0; m < n; m++) {
      var c = layout[m];
      c.vx += (cx - c.x) * kGrav;
      c.vy += (cy - c.y) * kGrav;
      c.vx *= 0.85; c.vy *= 0.85;
      c.x += c.vx * dt;
      c.y += c.vy * dt;
      // 拖拽中的节点固定
      if (m === dragIdx) { c.x = dragX; c.y = dragY; c.vx = 0; c.vy = 0; }
      // 边界软约束
      if (c.x < 20) c.x = 20;
      if (c.x > W - 20) c.x = W - 20;
      if (c.y < 20) c.y = H - 20;
      if (c.y > H - 20) c.y = H - 20;
    }
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    if (!layout.length) return;

    // hover 高亮集合
    var hl = {};
    if (hoverIdx >= 0 && layout[hoverIdx]) {
      var hv = layout[hoverIdx];
      hl[hv.id] = true;
      edges.forEach(function (e) {
        if (e.src === hv.id) hl[e.dst] = true;
        if (e.dst === hv.id) hl[e.src] = true;
      });
    }

    // 边
    edges.forEach(function (e) {
      var na = layoutById[e.src], nb = layoutById[e.dst];
      if (!na || !nb) return;
      var active = hl[e.src] && hl[e.dst];
      ctx.strokeStyle = active ? "rgba(148,216,195,0.65)" : "rgba(255,255,255,0.10)";
      ctx.lineWidth = active ? 1.6 : 1;
      ctx.beginPath();
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
      ctx.stroke();
    });

    // 节点
    layout.forEach(function (nd, i) {
      var isHover = i === hoverIdx;
      var isHL = hl[nd.id] && !isHover;
      ctx.globalAlpha = isHover ? 1 : (isHL ? 0.9 : 0.78);
      ctx.fillStyle = nd.color;
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, isHover ? nd.r + 3 : nd.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // 边框
      ctx.strokeStyle = isHover ? "#fff" : "rgba(255,255,255,0.25)";
      ctx.lineWidth = isHover ? 1.5 : 1;
      ctx.stroke();

      // 文字：领域常显；hover 节点 / 大节点显示
      if (nd.type === "topic" || isHover || (nd.type === "doc" && nd.deg >= 6)) {
        ctx.fillStyle = isHover ? "#ffffff" : "rgba(230,236,245,0.85)";
        ctx.font = (isHover ? "600 " : "") + "11px 'Microsoft YaHei', sans-serif";
        ctx.fillText(nd.name, nd.x + nd.r + 5, nd.y + 4);
      }
    });
  }

  /* ==================== 交互 ==================== */

  var dragX = 0, dragY = 0;

  function canvasPos(ev) {
    var rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }

  function pick(px, py) {
    var best = -1, bestD = 16;
    for (var i = 0; i < layout.length; i++) {
      var nd = layout[i];
      var d = Math.hypot(nd.x - px, nd.y - py);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function onDown(ev) {
    var p = canvasPos(ev);
    var i = pick(p.x, p.y);
    if (i >= 0) {
      dragIdx = i;
      dragX = layout[i].x; dragY = layout[i].y;
      canvas.classList.add("dragging");
      ev.preventDefault();
    }
  }

  function onMove(ev) {
    var p = canvasPos(ev);
    if (dragIdx >= 0) {
      dragX = p.x; dragY = p.y;
      return;
    }
    var i = pick(p.x, p.y);
    hoverIdx = i;
    if (i >= 0) {
      var nd = layout[i];
      tip.style.display = "block";
      tip.style.left = Math.min(p.x + 14, W - 270) + "px";
      tip.style.top = Math.max(p.y - 10, 6) + "px";
      tip.innerHTML =
        '<div class="t-name">' + esc(nd.name) + "</div>" +
        '<div class="t-sub">' + TYPE_LABEL[nd.type] +
        (nd.topic ? " · " + esc(nd.topic) : "") +
        " · 关联 " + nd.deg + "</div>";
    } else {
      tip.style.display = "none";
    }
  }

  function onUp() {
    dragIdx = -1;
    canvas.classList.remove("dragging");
  }

  function rebuildGraph(withEntities) {
    var btn = withEntities ? 2 : 1;
    setStatus("构建中…");
    toast("图谱重建已启动" + (withEntities ? "（含 LLM 实体抽取，需数分钟）" : "（元数据层，秒级）"), "success");
    fetch("/api/graph/rebuild", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ extract_entities: withEntities }),
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        if (!data.ok) { setStatus(""); return; }
        pollStatus();
      })
      .catch(function (e) { toast("启动失败: " + e, "error"); setStatus(""); });
  }

  function pollStatus() {
    fetch("/api/graph/status")
      .then(function (r) { return r.json(); })
      .then(function (s) {
        if (s.running) {
          setStatus("构建中… " + (s.stage || ""));
          setTimeout(pollStatus, 1200);
        } else if (s.error) {
          setStatus("");
          toast("图谱构建失败: " + s.error, "error");
        } else {
          setStatus("构建完成");
          toast("图谱构建完成", "success");
          loadGraph();
        }
      })
      .catch(function () { setStatus(""); });
  }

  function setStatus(t) {
    var el = $("kgStatus");
    if (el) el.textContent = t || "";
  }

  // 暴露给页面按钮
  window.loadGraph = loadGraph;
  window.rebuildGraph = rebuildGraph;

  document.addEventListener("DOMContentLoaded", init);
  if (document.readyState === "complete" || document.readyState === "interactive") {
    setTimeout(init, 0);
  }
})();
