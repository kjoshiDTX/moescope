import { useRef, useState } from "react";
import * as d3 from "d3";
import { getJSON } from "../api.js";
import { Card, DataTable, Empty, Select, StatTile, fmt, useAsync, useTip, useWidth } from "../charts.jsx";

export default function PruneView({ run }) {
  const all = useAsync(() => getJSON("/api/prune"), []);
  if (all.loading) return <div className="text-[13px] text-ink-2">Loading prune results…</div>;
  const mine = (all.data ?? []).filter((p) => p.name === run.name || p.name.startsWith(`${run.name}/`) || p.source_run === run.name.split("/").at(-1));
  if (!mine.length)
    return (
      <Empty command={`moescope prune ${run.name} --keep-coverage 90,95,99 --eval examples/code_eval.jsonl`}>
        No prune results for this run yet.
      </Empty>
    );
  return <Body results={mine} />;
}

function Body({ results }) {
  const [which, setWhich] = useState(results[0].name);
  const p = results.find((r) => r.name === which) ?? results[0];
  const keys = p.results.map((r) => String(r.keep_coverage));
  const [example, setExample] = useState(keys.at(-1));
  const best = [...p.results].sort((a, b) => b.masked_share - a.masked_share).find((r) => r.score >= p.baseline.score) ?? null;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label={`Full model (${p.metric})`} value={fmt.pct0(p.baseline.score)} note={`${p.baseline.n} eval examples`} />
        {p.results.slice(-3).map((r) => (
          <StatTile key={r.keep_coverage} label={`Keep Cov@${r.keep_coverage}`} value={fmt.pct0(r.score)}
            note={`${fmt.pct0(r.masked_share)} of experts masked · ${fmt.bytes(r.masked_bytes)}`} />
        ))}
      </div>
      <Card
        testid="view-prune"
        title="Quality vs experts masked"
        subtitle={`Experts outside each coverage set are masked in the router (logits set to −∞), then the eval reruns. Masking shows potential savings; the weights are still loaded. ${best ? `Keeping Cov@${best.keep_coverage} masks ${fmt.pct0(best.masked_share)} with no score loss.` : ""}`}
        actions={results.length > 1 && <Select label="Result" value={which} onChange={setWhich} options={results.map((r) => ({ value: r.name, label: r.name }))} />}
      >
        <TradeoffChart prune={p} />
      </Card>
      <Card
        title="Eval examples"
        subtitle="Model outputs with every expert available vs pruned."
        actions={<Select label="Pruned at" value={example} onChange={setExample} options={keys.map((k) => ({ value: k, label: `Cov@${k}` }))} />}
      >
        <DataTable
          testid="prune-examples"
          columns={[
            { key: "prompt", label: "Prompt", align: "left" },
            { key: "answer", label: "Expected", align: "left" },
            { key: "outputs", label: "Full", align: "left", format: (o, r) => <Mark ok={r.scores.full > 0}>{o.full}</Mark> },
            { key: "id", label: `Cov@${example}`, align: "left", format: (_, r) => <Mark ok={r.scores[example] > 0}>{r.outputs[example]}</Mark> },
          ]}
          rows={p.examples}
        />
      </Card>
    </div>
  );
}

function Mark({ ok, children }) {
  return (
    <span className="inline-flex items-start gap-1.5">
      <span aria-label={ok ? "correct" : "wrong"} className="mt-px font-semibold" style={{ color: ok ? "var(--good)" : "var(--critical)" }}>{ok ? "✓" : "✗"}</span>
      <span className="line-clamp-2 max-w-xs">{children}</span>
    </span>
  );
}

function TradeoffChart({ prune }) {
  const wrap = useRef(null);
  const width = useWidth(wrap);
  const { tip, ref, show, hide } = useTip();
  const height = 240;
  const m = { top: 16, right: 24, bottom: 40, left: 54 };
  const iw = Math.max(0, width - m.left - m.right);
  const ih = height - m.top - m.bottom;
  const pts = [{ label: "Full", masked_share: 0, score: prune.baseline.score }, ...prune.results.map((r) => ({ ...r, label: `Cov@${r.keep_coverage}` }))];
  const x = d3.scaleLinear().domain([0, Math.max(0.1, d3.max(pts, (d) => d.masked_share) * 1.15)]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 1]).range([ih, 0]);
  const sorted = [...pts].sort((a, b) => a.masked_share - b.masked_share);
  return (
    <div ref={ref} className="relative" data-testid="prune-chart">
      <div ref={wrap}>
        {width > 0 && (
          <svg width={width} height={height} role="img">
            <g transform={`translate(${m.left},${m.top})`}>
              {y.ticks(5).map((t) => (
                <g key={t} transform={`translate(0,${y(t)})`}><line x2={iw} stroke="var(--grid)" /><text x={-8} dy="0.32em" textAnchor="end">{fmt.pct0(t)}</text></g>
              ))}
              <g transform={`translate(0,${ih})`}>
                <line x2={iw} stroke="var(--axis)" />
                {x.ticks(5).map((t) => <text key={t} x={x(t)} y={16} textAnchor="middle">{fmt.pct0(t)}</text>)}
                <text x={iw / 2} y={34} textAnchor="middle">Share of routed experts masked</text>
              </g>
              <text transform={`translate(-40,${ih / 2}) rotate(-90)`} textAnchor="middle">Eval score</text>
              <line x1={0} x2={iw} y1={y(prune.baseline.score)} y2={y(prune.baseline.score)} stroke="var(--axis)" />
              <path d={d3.line().x((d) => x(d.masked_share)).y((d) => y(d.score))(sorted)} fill="none" stroke="var(--series-1)" strokeWidth={2} />
              {pts.map((d) => (
                <g key={d.label}>
                  <circle cx={x(d.masked_share)} cy={y(d.score)} r={5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
                  <text x={x(d.masked_share) + (d.masked_share === 0 ? 6 : 0)} y={y(d.score) - 10} textAnchor={d.masked_share === 0 ? "start" : "middle"} style={{ fill: "var(--ink-2)" }}>{d.label}</text>
                  <circle cx={x(d.masked_share)} cy={y(d.score)} r={14} fill="transparent"
                    onMouseMove={(e) => show(e, `${d.label}: score ${fmt.pct(d.score)} with ${fmt.pct(d.masked_share)} of experts masked`)} onMouseLeave={hide} />
                </g>
              ))}
            </g>
          </svg>
        )}
      </div>
      {tip && (
        <div role="tooltip" className="pointer-events-none absolute z-10 max-w-[220px] rounded-lg border border-line bg-surface px-3 py-2 text-[12px] text-ink shadow-lg" style={{ left: Math.min(tip.x + 14, width - 230), top: tip.y + 14 }}>
          {tip.content}
        </div>
      )}
    </div>
  );
}
