import { useState } from "react";
import * as d3 from "d3";
import { getJSON, runPath } from "../api.js";
import { Card, ChartTableToggle, DataTable, Empty, GradientLegend, Matrix, Select, StatTile, fmt, sequential, useAsync, useDark } from "../charts.jsx";

export function useReport(run) {
  return useAsync(() => getJSON(`${runPath(run.name)}/report`), [run.name]);
}

export const layerColumns = [
  { key: "layer", label: "Layer" },
  { key: "tokens", label: "Tokens", format: fmt.num },
  { key: "active_experts", label: "Active experts" },
  { key: "entropy_norm", label: "Entropy (norm.)", format: fmt.dec2 },
  { key: "coverage_90", label: "Cov@90" },
  { key: "coverage_95", label: "Cov@95" },
  { key: "coverage_99", label: "Cov@99" },
  { key: "logit_max_mean", label: "Mean max logit", format: (v) => (v == null ? "–" : fmt.dec2(v)) },
];

export function ReportState({ report, children }) {
  if (report.loading) return <div className="text-[13px] text-ink-2">Loading report…</div>;
  if (report.error) return <Empty>Could not load the report: {report.error.message}</Empty>;
  return children(report.data);
}

export default function UtilizationView({ run }) {
  const report = useReport(run);
  return <ReportState report={report}>{(r) => <Body report={r} />}</ReportState>;
}

function Body({ report }) {
  const dark = useDark();
  const [mode, setMode] = useState("chart");
  const [slice, setSlice] = useState("all");
  const [metric, setMetric] = useState("utilization");
  const m = report.manifest;

  const slices = [{ value: "all", label: "All tokens" }];
  for (const p of Object.keys(report.by_phase)) slices.push({ value: `phase:${p}`, label: `Phase: ${p}` });
  for (const c of Object.keys(report.by_category)) slices.push({ value: `cat:${c}`, label: `Category: ${c}` });

  let matrix = metric === "weighted" ? report.weighted_utilization : report.utilization;
  if (slice.startsWith("phase:")) matrix = report.by_phase[slice.slice(6)].utilization;
  if (slice.startsWith("cat:")) matrix = report.by_category[slice.slice(4)].utilization;
  const weighted = metric === "weighted" && slice === "all";
  const max = d3.max(matrix.flat()) || 1;
  const color = sequential(max, dark);
  const fair = weighted ? 1 / m.num_experts : m.top_k / m.num_experts;
  const s = report.summary;

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile testid="stat-tokens" label="Tokens profiled" value={fmt.num(report.tokens)} note={`${report.prompts} prompts · ${fmt.num(report.phases.decode)} decoded`} />
        <StatTile label="Experts for 95% of routing" value={fmt.pct0(s.coverage_95_share)} note={`Cov@90 ${fmt.pct0(s.coverage_90_share)} · Cov@99 ${fmt.pct0(s.coverage_99_share)}`} />
        <StatTile label="Never used" value={fmt.pct0(s.never_used_share)} note={`of ${m.num_layers * m.num_experts} routed experts`} />
        <StatTile label="Routing evenness" value={fmt.dec2(s.mean_entropy_norm)} note="mean normalized entropy · 1 = even" />
      </div>
      <Card
        testid="view-utilization"
        title="Expert utilization by layer"
        subtitle={`Share of tokens routed to each expert. With top-${m.top_k} of ${m.num_experts}, an even split puts every cell at ${fmt.dec(fair)}${weighted ? " (weighted by routing weight, so rows sum to 1)" : ""}.`}
        actions={
          <>
            <Select label="Tokens" value={slice} onChange={setSlice} options={slices} testid="slice-select" />
            <Select label="Metric" value={metric} onChange={setMetric}
              options={[{ value: "utilization", label: "Tokens routed" }, { value: "weighted", label: "Weighted by routing weight" }]} />
            <ChartTableToggle value={mode} onChange={setMode} />
          </>
        }
      >
        {mode === "chart" ? (
          <>
            <div className="mb-2"><GradientLegend color={color} max={max} format={fmt.dec} label="Utilization" /></div>
            <Matrix
              testid="heatmap"
              values={matrix}
              color={color}
              rowTitle="MoE layer"
              colTitle="Expert"
              tooltip={(r, c, v) => (
                <div>
                  <div className="font-medium">Layer {r} · expert {c}</div>
                  <div className="tabular text-ink-2">utilization {fmt.dec(v)} · {(v / fair).toFixed(2)}× an even share</div>
                </div>
              )}
            />
          </>
        ) : (
          <DataTable testid="utilization-table" columns={layerColumns} rows={report.layers} />
        )}
      </Card>
      <Card title="Hottest and coldest experts" subtitle="Per layer, by share of routed tokens.">
        <DataTable
          columns={[
            { key: "layer", label: "Layer" },
            { key: "hottest", label: "Hottest 10", align: "left", format: (v) => v.map((x) => x.expert).join(" ") },
            { key: "coldest", label: "Coldest 10", align: "left", format: (v) => v.map((x) => x.expert).join(" ") },
          ]}
          rows={report.hot_cold}
          maxHeight={280}
        />
      </Card>
    </div>
  );
}
