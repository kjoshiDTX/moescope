import { useState } from "react";
import { Card, ChartTableToggle, ColumnChart, DataTable, LineChart, Segmented, fmt } from "../charts.jsx";
import { ReportState, layerColumns, useReport } from "./Utilization.jsx";

export default function CoverageView({ run }) {
  const report = useReport(run);
  return <ReportState report={report}>{(r) => <Body report={r} />}</ReportState>;
}

function Body({ report }) {
  const [layer, setLayer] = useState(0);
  const [p, setP] = useState("95");
  const [mode, setMode] = useState("chart");
  const E = report.manifest.num_experts;
  const curves = report.coverage_curve;

  const series = curves.map((curve, l) => ({
    key: `L${l}`,
    label: `Layer ${l}`,
    color: "var(--series-1)",
    muted: l !== layer,
    points: [{ x: 0, y: 0 }, ...curve.map((y, i) => ({ x: i + 1, y }))],
  }));
  // Muted layers first so the selected one draws on top.
  series.sort((a, b) => Number(!a.muted) - Number(!b.muted));
  const columns = report.layers.map((row) => ({ x: row.layer, y: row[`coverage_${p}`] }));

  return (
    <div className="flex flex-col gap-4">
      <Card
        testid="view-coverage"
        title={`Coverage curve · layer ${layer}`}
        subtitle="Take experts hottest-first and add up their share of routed tokens. A curve that rises fast means a few experts carry the layer. Gray lines are the other layers; pick one below."
        actions={<ChartTableToggle value={mode} onChange={setMode} />}
      >
        {mode === "chart" ? (
          <LineChart
            testid="coverage-chart"
            series={series}
            xFormat={(v) => `${v}`}
            xLabel="Experts kept (hottest first)"
            yLabel="Share of routed tokens"
            yDomain={[0, 1]}
            markers={false}
            refLines={[{ y: 0.9, label: "90%" }, { y: 0.95, label: "95%" }, { y: 0.99, label: "99%" }]}
            tooltip={(x, rows) => (
              <div>
                <div className="font-medium">Top {x} of {E} experts</div>
                {rows.map(({ s, p }) => <div key={s.key} className="tabular text-ink-2">{s.label}: {fmt.pct(p.y)} of tokens</div>)}
              </div>
            )}
          />
        ) : (
          <DataTable columns={layerColumns} rows={report.layers} />
        )}
      </Card>
      <Card
        title={`Experts needed for ${p}% of routing, per layer`}
        subtitle="Click a column to show that layer's curve above."
        actions={<Segmented value={p} onChange={setP} options={["90", "95", "99"].map((v) => ({ value: v, label: `Cov@${v}` }))} testid="coverage-p" />}
      >
        <ColumnChart
          testid="coverage-columns"
          data={columns}
          yMax={E}
          xLabel="MoE layer"
          yLabel="Experts"
          selected={layer}
          onSelect={setLayer}
          tooltip={(d) => `Layer ${d.x}: ${d.y} of ${E} experts (${fmt.pct0(d.y / E)}) reach ${p}%`}
        />
      </Card>
    </div>
  );
}
