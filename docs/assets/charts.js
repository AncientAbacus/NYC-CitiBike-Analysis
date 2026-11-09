// Small, dependency-free SVG chart helpers shared by the page's sections.
// Every chart: one y-axis, recessive grid, legend for >= 2 series, hover
// tooltip (crosshair on lines, nearest point on scatter, per-bar on bars),
// and a repaint hook so light/dark colors are re-selected per mode.
(function () {
  const NS = "http://www.w3.org/2000/svg";

  function isDark() {
    const stamp = document.documentElement.getAttribute("data-theme");
    if (stamp === "dark") return true;
    if (stamp === "light") return false;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function onSchemeChange(fn) {
    if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fn);
  }
  // colors are {light, dark} pairs, each validated against its own surface
  function pick(c) { return typeof c === "string" ? c : (isDark() ? c.dark : c.light); }

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  // round tick step (1/2/2.5/5 x 10^k) so ticks land on readable values
  function niceScale(maxValue, target) {
    const raw = Math.max(maxValue, 1e-9) / (target || 4);
    const p = Math.pow(10, Math.floor(Math.log10(raw)));
    const n = raw / p;
    const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10) * p;
    const ticks = Math.max(1, Math.ceil(maxValue / step));
    return { max: ticks * step, ticks };
  }

  function fmtNum(v, digits) {
    if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString();
    return v.toLocaleString(undefined, { maximumFractionDigits: digits == null ? 1 : digits });
  }

  // ---- tooltip (one shared element, filled with textContent only) ----
  function tooltipEl() { return document.getElementById("tooltip"); }
  function showTip(evt, title, rows) {
    const tip = tooltipEl();
    tip.textContent = "";
    const b = document.createElement("b");
    b.textContent = title;
    tip.appendChild(b);
    rows.forEach(r => {
      const line = document.createElement("div");
      line.className = "tip-row";
      if (r.color) {
        const sw = document.createElement("span");
        sw.className = "tip-swatch";
        sw.style.background = r.color;
        line.appendChild(sw);
      }
      const val = document.createElement("strong");
      val.textContent = r.value;
      line.appendChild(val);
      line.appendChild(document.createTextNode(" " + r.label));
      tip.appendChild(line);
    });
    tip.style.left = evt.clientX + "px";
    tip.style.top = evt.clientY + "px";
    tip.classList.add("visible");
  }
  function hideTip() { tooltipEl().classList.remove("visible"); }

  function legend(container, items) {
    const row = document.createElement("div");
    row.className = "chart-legend";
    items.forEach(it => {
      const item = document.createElement("span");
      item.className = "chart-legend-item";
      const sw = document.createElement("span");
      sw.className = "chart-legend-swatch" + (it.kind === "dot" ? " dot" : "") + (it.dash ? " dash" : "");
      sw.style.setProperty("--swatch", pick(it.color));
      item.appendChild(sw);
      item.appendChild(document.createTextNode(it.name));
      row.appendChild(item);
    });
    container.appendChild(row);
  }

  // draw at the container's real width so text stays a constant size
  function frame(container, opts) {
    const W = Math.max(280, Math.round(container.clientWidth || 720)), H = opts.height || 260;
    const M = Object.assign({ top: 12, right: 16, bottom: 30, left: 52 }, opts.margin || {});
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": opts.ariaLabel || "" }, container);
    return { svg, W, H, M, iw: W - M.left - M.right, ih: H - M.top - M.bottom };
  }

  function yAxis(f, yMax, opts, ticks) {
    for (let i = 0; i <= ticks; i++) {
      const v = (yMax / ticks) * i;
      const y = f.M.top + f.ih - (v / yMax) * f.ih;
      el("line", { x1: f.M.left, x2: f.M.left + f.iw, y1: y, y2: y, class: i === 0 ? "chart-baseline" : "chart-grid" }, f.svg);
      const t = el("text", { x: f.M.left - 8, y: y + 4, class: "chart-tick", "text-anchor": "end" }, f.svg);
      t.textContent = opts.yFormat ? opts.yFormat(v) : fmtNum(v, 1);
    }
    if (opts.yLabel) {
      const t = el("text", { x: 12, y: f.M.top + f.ih / 2, class: "chart-axis-label", "text-anchor": "middle",
        transform: `rotate(-90 12 ${f.M.top + f.ih / 2})` }, f.svg);
      t.textContent = opts.yLabel;
    }
  }

  function xTicks(f, ticks, sx) {
    let last = -Infinity;
    ticks.forEach(t => {
      const x = sx(t.x);
      if (x - last < 34) return;  // drop labels that would collide on narrow screens
      last = x;
      el("line", { x1: x, x2: x, y1: f.M.top + f.ih, y2: f.M.top + f.ih + 4, class: "chart-baseline" }, f.svg);
      const txt = el("text", { x, y: f.M.top + f.ih + 18, class: "chart-tick", "text-anchor": "middle" }, f.svg);
      txt.textContent = t.label;
    });
  }

  // ---- line chart: series share one x array; crosshair + all-series tooltip ----
  // opts: { x: [..], series: [{name, color, values, dash, width}], xTicks: [{x,label}],
  //         xTitle(i) -> string, yFormat, yLabel, yMax, height, ariaLabel, valueFormat }
  function lineChart(container, opts) {
    function draw() {
      container.textContent = "";
      if (opts.series.length > 1) legend(container, opts.series);
      const f = frame(container, opts);
      const xs = opts.x;
      const xMin = xs[0], xMax = xs[xs.length - 1];
      const allY = opts.series.flatMap(s => s.values.filter(v => v != null));
      const sc = niceScale(Math.max(...allY), opts.yTicks), yMax = sc.max;
      const sx = x => f.M.left + ((x - xMin) / (xMax - xMin || 1)) * f.iw;
      const sy = y => f.M.top + f.ih - (y / yMax) * f.ih;
      yAxis(f, yMax, opts, sc.ticks);
      xTicks(f, opts.xTicks || [], sx);

      opts.series.forEach(s => {
        let d = "";
        s.values.forEach((v, i) => {
          if (v == null) return;
          d += (d && s.values[i - 1] != null ? "L" : "M") + sx(xs[i]).toFixed(1) + " " + sy(v).toFixed(1);
        });
        el("path", { d, fill: "none", stroke: pick(s.color), "stroke-width": s.width || 2,
          "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": s.dash || "none",
          opacity: s.opacity || 1 }, f.svg);
      });

      const hair = el("line", { y1: f.M.top, y2: f.M.top + f.ih, class: "chart-hair", visibility: "hidden" }, f.svg);
      const dots = opts.series.map(s => el("circle", { r: 4, fill: pick(s.color), class: "chart-dot", visibility: "hidden" }, f.svg));
      const hit = el("rect", { x: f.M.left, y: f.M.top, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
      function move(evt) {
        const pt = f.svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        const p = pt.matrixTransform(f.svg.getScreenCTM().inverse());
        const xv = xMin + ((p.x - f.M.left) / f.iw) * (xMax - xMin);
        let i = 0, best = Infinity;
        xs.forEach((x, j) => { const dd = Math.abs(x - xv); if (dd < best) { best = dd; i = j; } });
        const X = sx(xs[i]);
        hair.setAttribute("x1", X); hair.setAttribute("x2", X); hair.setAttribute("visibility", "visible");
        opts.series.forEach((s, k) => {
          const v = s.values[i];
          if (v == null) { dots[k].setAttribute("visibility", "hidden"); return; }
          dots[k].setAttribute("cx", X); dots[k].setAttribute("cy", sy(v)); dots[k].setAttribute("visibility", "visible");
        });
        showTip(evt, opts.xTitle ? opts.xTitle(i) : String(xs[i]), opts.series
          .filter(s => s.values[i] != null)
          .map(s => ({ color: pick(s.color), label: s.name, value: (opts.valueFormat || fmtNum)(s.values[i]) })));
      }
      hit.addEventListener("pointermove", move);
      hit.addEventListener("pointerleave", () => {
        hideTip(); hair.setAttribute("visibility", "hidden"); dots.forEach(d => d.setAttribute("visibility", "hidden"));
      });
    }
    draw();
    onSchemeChange(draw);
    redrawers.push(draw);
  }

  // ---- scatter with optional overlay line; nearest-point hover ----
  // opts: { groups: [{name, color, points: [{x, y, title, rows}]}], line: {name, color, points: [{x,y}]},
  //         xMin, xMax, xTicks, yFormat, yLabel, xLabel, height, ariaLabel }
  function scatterChart(container, opts) {
    function draw() {
      container.textContent = "";
      const items = opts.groups.map(g => ({ name: g.name, color: g.color, kind: "dot" }));
      if (opts.line) items.push({ name: opts.line.name, color: opts.line.color, dash: opts.line.dash });
      legend(container, items);
      const f = frame(container, Object.assign({ margin: { bottom: 40 } }, opts));
      const pts = opts.groups.flatMap(g => g.points);
      const sc = niceScale(Math.max(...pts.map(p => p.y)), opts.yTicks), yMax = sc.max;
      const sx = x => f.M.left + ((x - opts.xMin) / (opts.xMax - opts.xMin)) * f.iw;
      const sy = y => f.M.top + f.ih - (y / yMax) * f.ih;
      yAxis(f, yMax, opts, sc.ticks);
      xTicks(f, opts.xTicks || [], sx);
      if (opts.xLabel) {
        const t = el("text", { x: f.M.left + f.iw / 2, y: f.H - 4, class: "chart-axis-label", "text-anchor": "middle" }, f.svg);
        t.textContent = opts.xLabel;
      }
      const drawn = [];
      opts.groups.forEach(g => {
        g.points.forEach(p => {
          const c = el("circle", { cx: sx(p.x), cy: sy(p.y), r: 3.5, fill: pick(g.color), class: "chart-scatter-dot" }, f.svg);
          drawn.push({ p, c, g });
        });
      });
      if (opts.line) {
        const d = opts.line.points.map((p, i) => (i ? "L" : "M") + sx(p.x).toFixed(1) + " " + sy(p.y).toFixed(1)).join("");
        el("path", { d, fill: "none", stroke: pick(opts.line.color), "stroke-width": 2.5, "stroke-dasharray": opts.line.dash || "none" }, f.svg);
      }
      const ring = el("circle", { r: 6, class: "chart-ring", visibility: "hidden" }, f.svg);
      const hit = el("rect", { x: f.M.left, y: f.M.top, width: f.iw, height: f.ih, fill: "transparent" }, f.svg);
      hit.addEventListener("pointermove", evt => {
        const pt = f.svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        const q = pt.matrixTransform(f.svg.getScreenCTM().inverse());
        let best = null, bd = Infinity;
        drawn.forEach(o => {
          const dx = sx(o.p.x) - q.x, dy = sy(o.p.y) - q.y, dd = dx * dx + dy * dy;
          if (dd < bd) { bd = dd; best = o; }
        });
        if (!best || bd > 900) { hideTip(); ring.setAttribute("visibility", "hidden"); return; }
        ring.setAttribute("cx", sx(best.p.x)); ring.setAttribute("cy", sy(best.p.y)); ring.setAttribute("visibility", "visible");
        showTip(evt, best.p.title, best.p.rows);
      });
      hit.addEventListener("pointerleave", () => { hideTip(); ring.setAttribute("visibility", "hidden"); });
    }
    draw();
    onSchemeChange(draw);
    redrawers.push(draw);
  }

  // ---- vertical bars, one series; per-bar hover ----
  // opts: { bars: [{label, value, title}], color, yFormat, yLabel, height, ariaLabel, valueFormat }
  function barChart(container, opts) {
    function draw() {
      container.textContent = "";
      const f = frame(container, opts);
      const sc = niceScale(Math.max(...opts.bars.map(b => b.value)), opts.yTicks), yMax = sc.max;
      yAxis(f, yMax, opts, sc.ticks);
      const n = opts.bars.length, slot = f.iw / n, bw = Math.max(4, slot - 6);
      opts.bars.forEach((b, i) => {
        const x = f.M.left + i * slot + (slot - bw) / 2;
        const h = (b.value / yMax) * f.ih, y = f.M.top + f.ih - h;
        const r = Math.min(4, bw / 2, h);
        // rounded data-end, square at the baseline
        const d = `M${x} ${f.M.top + f.ih}V${y + r}Q${x} ${y} ${x + r} ${y}H${x + bw - r}Q${x + bw} ${y} ${x + bw} ${y + r}V${f.M.top + f.ih}Z`;
        const bar = el("path", { d, fill: pick(opts.color), class: "chart-bar" }, f.svg);
        const t = el("text", { x: x + bw / 2, y: f.M.top + f.ih + 18, class: "chart-tick", "text-anchor": "middle" }, f.svg);
        t.textContent = b.label;
        const hit = el("rect", { x: f.M.left + i * slot, y: f.M.top, width: slot, height: f.ih, fill: "transparent" }, f.svg);
        hit.addEventListener("pointermove", evt => {
          bar.classList.add("hover");
          showTip(evt, b.title || b.label, [{ label: opts.valueLabel || "", value: (opts.valueFormat || fmtNum)(b.value) }]);
        });
        hit.addEventListener("pointerleave", () => { bar.classList.remove("hover"); hideTip(); });
      });
    }
    draw();
    onSchemeChange(draw);
    redrawers.push(draw);
  }

  // charts redraw at their new width when the window is resized
  const redrawers = [];
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => redrawers.forEach(fn => fn()), 150);
  });

  window.Charts = { lineChart, scatterChart, barChart, isDark, pick, fmtNum, showTip, hideTip, onSchemeChange };
})();
