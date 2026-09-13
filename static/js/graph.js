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
  var TYPE_RADIUS = { doc: 9, topic: 16, tag: 6, entity: 5 };
  var TYPE_LABEL = { doc: "文档", topic: "领域", tag: "标签", entity: "实体" };

  var canvas, ctx, wrap, tip;
  var nodes = [], edges = [];
  // 默认只显示主干（文档/领域/标签）；实体层 1008 个节点按需勾选开启
  var visibleTypes = { doc: true, topic: true, tag: true, entity: false };
  var layout = [];           // {x, y, vx, vy, r, color, name, type, topic, deg}
  var layoutById = {};
  var W = 0, H = 0, dpr = 1;
  var raf = null, running = false;
  var dragIdx = -1, hoverIdx = -1;
  var settled = 0;

  function $(id) { return document.getElementById(id); }

  var inited = false;

  function init() {
    if (inited) return;   // DOMContentLoaded 与 readyState 双触发防重入
    inited = true;
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
        setStatus("当前显示 " + layout.length + " 节点 · " + edges.length + " 边");
      });
    });

    // 拖拽 / hover
    canvas.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    canvas.addEventListener("mouseleave", function () { hoverIdx = -1; });

    // 从后台切回 / 重新聚焦 → 立即重画（rAF 在后台被暂停，回来要恢复）
    window.addEventListener("focus", function () {
      if (layout.length && !running) settle(false);
    });
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && layout.length && !running) settle(false);
    });

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
        setStatus("当前显示 " + layout.length + " 节点 · " + edges.length +
                  " 边（实体层默认隐藏，勾选「实体」查看完整图谱）");
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
    // 分簇初始位置：领域节点放圆周 → 文档挂自己领域扇区 → 标签/实体放内环
    var rad = Math.min(W, H) * 0.36;
    var cx = W / 2, cy = H / 2;
    var topicAng = {};   // topic 名 → 扇区角度
    var topicIdx = 0, topicCount = 0;
    visNodes.forEach(function (nd) {
      if (nd.type === "topic") topicCount++;
    });
    function hashAng(s) {
      var h = 0;
      for (var i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) % 100000;
      return (h / 100000) * Math.PI * 2;
    }
    layout = visNodes.map(function (nd, i) {
      var p;
      if (nd.type === "topic") {
        // 领域：圆周均布
        var ang = (i / Math.max(topicCount, 1)) * Math.PI * 2 - Math.PI / 2;
        topicAng[nd.name] = ang;
        p = { x: cx + Math.cos(ang) * rad, y: cy + Math.sin(ang) * rad };
      } else if (nd.type === "doc" && nd.topic) {
        // 文档：挂到所属领域扇区附近（同一领域聚成一小簇）
        var a = (topicAng[nd.topic] !== undefined) ? topicAng[nd.topic] : hashAng(nd.topic || "");
        var rr = rad * (0.42 + Math.random() * 0.3);
        p = {
          x: cx + Math.cos(a + (Math.random() - 0.5) * 0.55) * rr,
          y: cy + Math.sin(a + (Math.random() - 0.5) * 0.55) * rr,
        };
      } else {
        // 标签 / 实体：内环随机分布
        var a2 = Math.random() * Math.PI * 2;
        var rr2 = rad * (0.12 + Math.random() * 0.3);
        p = { x: cx + Math.cos(a2) * rr2, y: cy + Math.sin(a2) * rr2 };
      }
      return {
        id: nd.id, name: nd.name, type: nd.type, topic: nd.topic,
        r: TYPE_RADIUS[nd.type] || 6, color: TYPE_COLOR[nd.type] || "#999",
        x: p.x, y: p.y,
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
      for (var i = 0; i < 200; i++) step(1);
      draw();
      return;
    }
    // 关键：先静默收敛 + 立即画一帧，保证首屏必有内容
    // （requestAnimationFrame 在无焦点/后台窗口会被暂停，动画可等，图不能等）
    for (var i = 0; i < 150; i++) step(1);
    draw();
    if (raf) cancelAnimationFrame(raf);
    running = true;
    var last = performance.now();
    // 帧数随规模自适应：节点越多动画越短，避免长时间满载卡交互
    var maxFrames = layout.length > 600 ? 350 : (layout.length > 300 ? 500 : 900);
    function loop(now) {
      var dt = Math.min(now - last, 50) / 16.7;   // 归一化到 60fps
      last = now;
      step(dt);
      draw();
      settled++;
      if (settled < maxFrames) {
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
    // dt 兜底：无参调用（同步预收敛）时按 1 帧推进，防止坐标变 NaN
    if (typeof dt !== "number" || !isFinite(dt)) dt = 1;
    var cx = W / 2, cy = H / 2;
    var kRep = 4200 * Math.sqrt(n) / Math.max(n, 10);   // 斥力强度随规模自适应
    var kSpr = 0.02, rest = 100;                        // 弹簧更长 → 连边拉开，簇更清晰
    var kGrav = 0.006;                                  // 中心引力弱 → 不被吸成一团
    var minD = 12;

    // 斥力：网格空间分区（只算同格 + 相邻 8 格），O(n²) → O(n·k)
    // 全量 1259 节点时每帧从 158 万对降到几万对，不再卡死主线程
    var cell = Math.max(70, minD * 4);
    var grid = {};
    for (var gi = 0; gi < n; gi++) {
      var gn = layout[gi];
      var gk = Math.floor(gn.x / cell) + "," + Math.floor(gn.y / cell);
      (grid[gk] = grid[gk] || []).push(gi);
    }
    for (var i = 0; i < n; i++) {
      var a = layout[i];
      var ax = Math.floor(a.x / cell), ay = Math.floor(a.y / cell);
      for (var ox = -1; ox <= 1; ox++) {
        for (var oy = -1; oy <= 1; oy++) {
          var bucket = grid[(ax + ox) + "," + (ay + oy)];
          if (!bucket) continue;
          for (var bi = 0; bi < bucket.length; bi++) {
            var j = bucket[bi];
            if (j <= i) continue;   // 每对只算一次
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

    // 边：普通边一次 path 批量画（全量 2202 条也只 2 次 stroke，不卡）
    var hasHL = false;
    ctx.strokeStyle = "rgba(255,255,255,0.055)";
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (var k = 0; k < edges.length; k++) {
      var e = edges[k];
      var na = layoutById[e.src], nb = layoutById[e.dst];
      if (!na || !nb) continue;
      if (hl[e.src] && hl[e.dst]) { hasHL = true; continue; }
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
    }
    ctx.stroke();
    // 高亮边单独一层（hover 时）
    if (hasHL) {
      ctx.strokeStyle = "rgba(148,216,195,0.7)";
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      for (var k2 = 0; k2 < edges.length; k2++) {
        var e2 = edges[k2];
        var na2 = layoutById[e2.src], nb2 = layoutById[e2.dst];
        if (!na2 || !nb2) continue;
        if (!(hl[e2.src] && hl[e2.dst])) continue;
        ctx.moveTo(na2.x, na2.y);
        ctx.lineTo(nb2.x, nb2.y);
      }
      ctx.stroke();
    }

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

      // 文字：领域常显；hover 节点 / 高关联文档（deg>=8）显示，避免文字打架
      if (nd.type === "topic" || isHover || (nd.type === "doc" && nd.deg >= 8)) {
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
      if (!running) draw();   // 动画已停：拖拽即时重绘，不跑物理
      return;
    }
    var i = pick(p.x, p.y);
    if (i !== hoverIdx) {
      hoverIdx = i;
      if (!running) draw();   // 动画已停：hover 即时高亮
    }
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
    if (dragIdx >= 0 && !running) settle(false);  // 松手后布局重新收敛
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
