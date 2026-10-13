import { useEffect, useLayoutEffect, useRef, useState } from "react";
import * as d3 from "d3";

// ---- theme & color ------------------------------------------------------------------------
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];
const RED = ["#f9d4d3", "#f2a9a8", "#ea7d7c", "#e34948", "#c23433", "#9b2423"];

export function useDark() {
  const query = () =>
    document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  const [dark, setDark] = useState(query);
  useEffect(() => {
    const update = () => setDark(query());
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => { mq.removeEventListener("change", update); obs.disconnect(); };
  }, []);
  return dark;
}

/** One hue, near-surface at zero, dark (light in dark mode) at max. */
export function sequential(max, dark) {
  const ramp = dark ? [...BLUE].reverse().slice(1) : BLUE;
  const interp = d3.interpolateRgbBasis(ramp);
  return (v) => interp(max > 0 ? Math.max(0, Math.min(1, v / max)) : 0);
}

/** Blue below zero, red above, neutral gray at zero; symmetric extent. */
export function diverging(extent, dark) {
  const mid = dark ? "#383835" : "#f0efec";
  const neg = d3.interpolateRgbBasis([mid, ...(dark ? ["#184f95", "#3987e5", "#86b6ef"] : ["#86b6ef", "#2a78d6", "#0d366b"])]);
  const pos = d3.interpolateRgbBasis([mid, ...(dark ? ["#9b2423", "#e34948", "#f2a9a8"] : ["#f2a9a8", "#e34948", "#9b2423"])]);
  return (v) => {
    const t = extent > 0 ? Math.max(-1, Math.min(1, v / extent)) : 0;
    return t < 0 ? neg(-t) : pos(t);
  };
}

export const fmt = {
  pct: d3.format(".1%"),
  pct0: d3.format(".0%"),
  num: d3.format(",.0f"),
  dec: d3.format(".3f"),
  dec2: d3.format(".2f"),
  bytes: (b) => (b >= 2 ** 30 ? `${(b / 2 ** 30).toFixed(2)} GiB` : `${(b / 2 ** 20).toFixed(1)} MiB`),
};

// ---- layout primitives ----------------------------------------------------------------------
export function useWidth(ref) {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export function Card({ title, subtitle, actions, children, testid }) {
  return (
    <section data-testid={testid} className="rounded-xl border border-line bg-surface p-4 sm:p-5">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-0.5 max-w-3xl text-[13px] text-ink-2">{subtitle}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

export function Select({ label, value, onChange, options, testid }) {
  return (
    <label className="flex items-center gap-1.5 text-[13px] text-ink-2">
      {label}
      <select
        data-testid={testid}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-line bg-surface px-2 py-1 text-ink"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>{o.label}</option>
        ))}
      </select>
    </label>
  );
}

export function Segmented({ value, onChange, options, testid }) {
  return (
    <div data-testid={testid} className="inline-flex rounded-md border border-line p-0.5 text-[13px]">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`rounded px-2.5 py-0.5 ${value === o.value ? "bg-wash font-medium text-ink" : "text-ink-2 hover:text-ink"}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function ChartTableToggle({ value, onChange }) {
  return <Segmented value={value} onChange={onChange} options={[{ value: "chart", label: "Chart" }, { value: "table", label: "Table" }]} />;
}

export function StatTile({ label, value, note, testid }) {
  return (
    <div data-testid={testid} className="rounded-xl border border-line bg-surface px-4 py-3">
      <div className="text-[12px] text-ink-2">{label}</div>
      <div className="mt-0.5 text-[22px] font-semibold text-ink">{value}</div>
      {note && <div className="text-[12px] text-muted">{note}</div>}
    </div>
  );
}

export function Empty({ children, command }) {
  return (
    <div className="rounded-xl border border-dashed border-line px-6 py-10 text-center text-[14px] text-ink-2">
      <p>{children}</p>
      {command && <code className="mt-2 inline-block rounded bg-wash px-2 py-1 font-mono text-[12px] text-ink">{command}</code>}
    </div>
  );
}

export function DataTable({ columns, rows, testid, maxHeight = 420 }) {
  return (
    <div data-testid={testid} className="overflow-auto rounded-lg border border-line" style={{ maxHeight }}>
      <table className="tabular w-full border-collapse text-[13px]">
        <thead className="sticky top-0 bg-surface">
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={`border-b border-line px-3 py-2 font-medium text-ink-2 ${c.align === "left" ? "text-left" : "text-right"}`}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line last:border-0">
              {columns.map((c) => (
                <td key={c.key} className={`px-3 py-1.5 text-ink ${c.align === "left" ? "text-left" : "text-right"}`}>
                  {c.format ? c.format(r[c.key], r) : r[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---- tooltip ------------------------------------------------------------------------------------
function Tooltip({ tip, width }) {
  if (!tip) return null;
  const left = Math.min(tip.x + 14, Math.max(0, width - 230));
  return (
    <div
      role="tooltip"
      className="pointer-events-none absolute z-10 w-max max-w-[220px] rounded-lg border border-line bg-surface px-3 py-2 text-[12px] text-ink shadow-lg"
      style={{ left, top: tip.y + 14 }}
    >
      {tip.content}
    </div>
  );
}

export function useTip() {
  const [tip, setTip] = useState(null);
  const ref = useRef(null);
  const show = (event, content) => {
    const box = ref.current.getBoundingClientRect();
    setTip({ x: event.clientX - box.left, y: event.clientY - box.top, content });
  };
  return { tip, ref, show, hide: () => setTip(null) };
}

export function Swatch({ color, line }) {
  return line ? (
    <span className="inline-block h-0.5 w-3 rounded align-middle" style={{ background: color }} />
  ) : (
    <span className="inline-block h-2.5 w-2.5 rounded-sm align-middle" style={{ background: color }} />
  );
}

export function Legend({ items }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-1 text-[12px] text-ink-2">
      {items.map((it) => (
        <span key={it.label} className="flex items-center gap-1.5">
          <Swatch color={it.color} line={it.line} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export function GradientLegend({ color, max, min = 0, format, label }) {
  const stops = d3.range(0, 1.0001, 0.1).map((t) => color(min + t * (max - min)));
  return (
    <div className="flex items-center gap-2 text-[12px] text-ink-2">
      {label && <span>{label}</span>}
      <span className="tabular">{format(min)}</span>
      <span className="h-2.5 w-32 rounded-sm" style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }} />
      <span className="tabular">{format(max)}</span>
    </div>
  );
}

/** Left margin wide enough for the longest y tick label (about 6.5px per character). */
function leftMargin(scale, format, ticks, hasLabel) {
  const widest = Math.max(...scale.ticks(ticks).map((t) => String(format(t)).length), 1);
  return Math.ceil(widest * 6.5) + 14 + (hasLabel ? 18 : 0);
}

/** Spread label y positions so none sit closer than `gap` px (keeps order). */
export function spreadLabels(items, gap = 13, maxY = Infinity) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  for (let i = 1; i < sorted.length; i++) sorted[i].y = Math.max(sorted[i].y, sorted[i - 1].y + gap);
  // If the stack ran past the bottom of the plot, push it back up.
  if (sorted.length && sorted.at(-1).y > maxY) {
    sorted.at(-1).y = maxY;
    for (let i = sorted.length - 2; i >= 0; i--) sorted[i].y = Math.min(sorted[i].y, sorted[i + 1].y - gap);
  }
  return sorted;
}

// ---- axes -----------------------------------------------------------------------------------
function YAxis({ scale, width, format, ticks = 5, label, labelOffset = 40 }) {
  return (
    <g>
      {scale.ticks(ticks).map((t) => (
        <g key={t} transform={`translate(0,${scale(t)})`}>
          <line x2={width} stroke="var(--grid)" />
          <text x={-8} dy="0.32em" textAnchor="end">{format(t)}</text>
        </g>
      ))}
      {label && <text transform={`translate(${-labelOffset},${(scale.range()[0] + scale.range()[1]) / 2}) rotate(-90)`} textAnchor="middle">{label}</text>}
    </g>
  );
}

function XAxis({ scale, height, format, ticks = 6, label, width, values }) {
  const tickValues = values ?? scale.ticks(ticks);
  return (
    <g transform={`translate(0,${height})`}>
      <line x2={width} stroke="var(--axis)" />
      {tickValues.map((t) => (
        <text key={t} x={scale(t)} y={16} textAnchor="middle">{format(t)}</text>
      ))}
      {label && <text x={width / 2} y={34} textAnchor="middle">{label}</text>}
    </g>
  );
}

// ---- line chart ---------------------------------------------------------------------------
/**
 * series: [{key, label, color, points: [{x, y}], muted?}]. Muted series draw as quiet context
 * and are left out of the legend and tooltip. refLines: [{y, label}].
 */
export function LineChart({ series, height = 260, xFormat = String, yFormat = fmt.pct0, yDomain, xLabel, yLabel, refLines = [], tooltip, testid, markers = true, xTicks }) {
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const { tip, ref, show, hide } = useTip();
  const [hoverX, setHoverX] = useState(null);
  const all = series.flatMap((s) => s.points);
  const xs = [...new Set(series.filter((s) => !s.muted).flatMap((s) => s.points.map((p) => p.x)))].sort((a, b) => a - b);
  const active = series.filter((s) => !s.muted);
  const endLabels = active.length > 1 && active.length <= 4;
  const m = { top: 12, right: endLabels || refLines.length ? 64 : 16, bottom: xLabel ? 40 : 26, left: 0 };
  const ih = height - m.top - m.bottom;
  const y = d3.scaleLinear().domain(yDomain ?? [0, d3.max(all, (p) => p.y) * 1.1 || 1]).nice().range([ih, 0]);
  m.left = leftMargin(y, yFormat, 5, !!yLabel);
  const iw = Math.max(0, width - m.left - m.right);
  const x = d3.scaleLinear().domain(d3.extent(all, (p) => p.x)).range([0, iw]);
  const xValues = xs.length && xs.length <= 12 ? xs : undefined;
  const sideLabels = spreadLabels([
    ...refLines.map((r) => ({ key: `ref-${r.label}`, y: y(r.y), text: r.label, color: "var(--muted)" })),
    ...(endLabels ? active.filter((s) => s.points.length).map((s) => ({ key: s.key, y: y(s.points.at(-1).y), text: s.label, color: "var(--ink-2)" })) : []),
  ], 13, ih - 6);
  const line = d3.line().x((p) => x(p.x)).y((p) => y(p.y)).curve(d3.curveMonotoneX);

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const px = x.invert(e.clientX - box.left);
    const nearest = xs.reduce((best, v) => (Math.abs(v - px) < Math.abs(best - px) ? v : best), xs[0]);
    setHoverX(nearest);
    const rows = active.map((s) => ({ s, p: s.points.find((p) => p.x === nearest) })).filter((r) => r.p);
    show(e, tooltip ? tooltip(nearest, rows) : (
      <div>
        <div className="mb-1 font-medium">{xFormat(nearest)}</div>
        {rows.map(({ s, p }) => (
          <div key={s.key} className="flex items-center gap-2"><Swatch color={s.color} line /> <span className="text-ink-2">{s.label}</span> <span className="tabular ml-auto">{yFormat(p.y)}</span></div>
        ))}
      </div>
    ));
  };

  return (
    <div ref={ref} className="relative" data-testid={testid}>
      {active.length > 1 && <Legend items={active.map((s) => ({ label: s.label, color: s.color, line: true }))} />}
      <div ref={wrap}>
        {width > 0 && (
          <svg width={width} height={height} role="img">
            <g transform={`translate(${m.left},${m.top})`}>
              <YAxis scale={y} width={iw} format={yFormat} label={yLabel} labelOffset={m.left - 12} />
              <XAxis scale={x} height={ih} width={iw} format={xFormat} label={xLabel} ticks={xTicks} values={xValues} />
              {refLines.map((r) => <line key={r.label} y1={y(r.y)} y2={y(r.y)} x2={iw} stroke="var(--axis)" />)}
              {sideLabels.map((l) => <text key={l.key} x={iw + 8} y={l.y} dy="0.32em" style={{ fill: l.color }}>{l.text}</text>)}
              {series.filter((s) => s.muted).map((s) => (
                <path key={s.key} d={line(s.points)} fill="none" stroke="var(--axis)" strokeWidth={1} />
              ))}
              {active.map((s) => (
                <g key={s.key}>
                  <path d={line(s.points)} fill="none" stroke={s.color} strokeWidth={2} strokeLinejoin="round" strokeLinecap="round" />
                  {markers && s.points.length <= 24 && s.points.map((p) => (
                    <circle key={p.x} cx={x(p.x)} cy={y(p.y)} r={4} fill={s.color} stroke="var(--surface)" strokeWidth={2} />
                  ))}
                </g>
              ))}
              {hoverX != null && <line x1={x(hoverX)} x2={x(hoverX)} y2={ih} stroke="var(--muted)" />}
              <rect width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={() => { hide(); setHoverX(null); }} />
            </g>
          </svg>
        )}
      </div>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ---- column chart -------------------------------------------------------------------------
export function ColumnChart({ data, height = 200, yFormat = String, yLabel, xLabel, color = "var(--series-1)", selected, onSelect, tooltip, yMax, testid, refLine }) {
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const { tip, ref, show, hide } = useTip();
  const m = { top: 10, right: 12, bottom: xLabel ? 40 : 26, left: 0 };
  const ih = height - m.top - m.bottom;
  const y = d3.scaleLinear().domain([0, yMax ?? (d3.max(data, (d) => d.y) || 1)]).nice().range([ih, 0]);
  m.left = leftMargin(y, yFormat, 4, !!yLabel);
  const iw = Math.max(0, width - m.left - m.right);
  const x = d3.scaleBand().domain(data.map((d) => d.x)).range([0, iw]).paddingInner(0.2);
  const bw = Math.min(24, x.bandwidth());
  const every = Math.ceil(data.length / Math.max(1, Math.floor(iw / 28)));
  const r = Math.min(4, bw / 2);
  const barPath = (x0, top, w, h) =>
    h <= 0 ? "" : `M${x0},${top + h}V${top + r}Q${x0},${top} ${x0 + r},${top}H${x0 + w - r}Q${x0 + w},${top} ${x0 + w},${top + r}V${top + h}Z`;

  return (
    <div ref={ref} className="relative" data-testid={testid}>
      <div ref={wrap}>
        {width > 0 && (
          <svg width={width} height={height} role="img">
            <g transform={`translate(${m.left},${m.top})`}>
              <YAxis scale={y} width={iw} format={yFormat} label={yLabel} ticks={4} labelOffset={m.left - 12} />
              <XAxis scale={(v) => x(v) + x.bandwidth() / 2} height={ih} width={iw} format={String} label={xLabel} values={data.map((d) => d.x).filter((_, i) => i % every === 0)} />
              {refLine && (
                <g transform={`translate(0,${y(refLine.y)})`}>
                  <line x2={iw} stroke="var(--critical)" strokeWidth={1} />
                  <text x={iw} y={-4} textAnchor="end" style={{ fill: "var(--critical)" }}>{refLine.label}</text>
                </g>
              )}
              {data.map((d) => {
                const x0 = x(d.x) + (x.bandwidth() - bw) / 2;
                const dim = selected != null && selected !== d.x;
                return (
                  <g key={d.x}>
                    <path d={barPath(x0, y(d.y), bw, ih - y(d.y))} fill={d.color ?? color} opacity={dim ? 0.35 : 1} />
                    <rect
                      x={x(d.x)} y={0} width={x.bandwidth()} height={ih} fill="transparent"
                      style={{ cursor: onSelect ? "pointer" : "default" }}
                      onMouseMove={(e) => show(e, tooltip ? tooltip(d) : `${d.x}: ${yFormat(d.y)}`)}
                      onMouseLeave={hide}
                      onClick={() => onSelect?.(d.x)}
                    />
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

// ---- matrix / heatmap -----------------------------------------------------------------------
/** values[r][c]; color(v) -> css color; null cells draw as an empty wash. */
export function Matrix({ values, color, rowLabel = (r) => r, colLabel = (c) => c, tooltip, rowTitle, colTitle, maxCell = 22, onCellClick, testid, highlightRow }) {
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const { tip, ref, show, hide } = useTip();
  const rows = values.length;
  const cols = values[0]?.length ?? 0;
  const m = { top: 4, right: 4, bottom: 30, left: 40 };
  const cell = Math.max(3, Math.min(maxCell, Math.floor((width - m.left - m.right) / Math.max(cols, 1))));
  const cellH = Math.max(4, Math.min(maxCell, cell));
  const gap = cell >= 8 ? 1 : 0;
  const iw = cell * cols;
  const ih = cellH * rows;
  const everyC = Math.ceil(24 / cell);
  const everyR = Math.ceil(14 / cellH);

  const onMove = (e) => {
    const box = e.currentTarget.getBoundingClientRect();
    const c = Math.floor((e.clientX - box.left) / cell);
    const r = Math.floor((e.clientY - box.top) / cellH);
    if (r < 0 || c < 0 || r >= rows || c >= cols) return hide();
    show(e, tooltip(r, c, values[r][c]));
  };

  return (
    <div ref={ref} className="relative" data-testid={testid}>
      <div ref={wrap}>
        {width > 0 && (
          <svg width={Math.max(iw + m.left + m.right, 0)} height={ih + m.top + m.bottom} role="img">
            <g transform={`translate(${m.left},${m.top})`}>
              {values.map((row, r) =>
                row.map((v, c) => (
                  <rect key={`${r}-${c}`} x={c * cell} y={r * cellH} width={cell - gap} height={cellH - gap}
                    fill={v == null ? "var(--wash)" : color(v)} rx={cell >= 10 ? 1.5 : 0} />
                )),
              )}
              {highlightRow != null && (
                <rect x={-2} y={highlightRow * cellH - 1} width={iw + 3} height={cellH + 1} fill="none" stroke="var(--ink)" strokeWidth={1.5} rx={2} />
              )}
              {d3.range(rows).filter((r) => r % everyR === 0).map((r) => (
                <text key={r} x={-6} y={r * cellH + cellH / 2} dy="0.32em" textAnchor="end">{rowLabel(r)}</text>
              ))}
              {d3.range(cols).filter((c) => c % everyC === 0).map((c) => (
                <text key={c} x={c * cell + cell / 2} y={ih + 13} textAnchor="middle">{colLabel(c)}</text>
              ))}
              {colTitle && <text x={iw / 2} y={ih + 27} textAnchor="middle">{colTitle}</text>}
              {rowTitle && <text transform={`translate(-30,${ih / 2}) rotate(-90)`} textAnchor="middle">{rowTitle}</text>}
              <rect width={iw} height={ih} fill="transparent" onMouseMove={onMove} onMouseLeave={hide}
                onClick={(e) => {
                  if (!onCellClick) return;
                  const box = e.currentTarget.getBoundingClientRect();
                  onCellClick(Math.floor((e.clientY - box.top) / cellH), Math.floor((e.clientX - box.left) / cell));
                }}
                style={{ cursor: onCellClick ? "pointer" : "default" }} />
            </g>
          </svg>
        )}
      </div>
      <Tooltip tip={tip} width={width} />
    </div>
  );
}

export function useAsync(fn, deps) {
  const [state, setState] = useState({ loading: true, data: null, error: null });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    fn().then(
      (data) => live && setState({ loading: false, data, error: null }),
      (error) => live && setState({ loading: false, data: null, error }),
    );
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}
