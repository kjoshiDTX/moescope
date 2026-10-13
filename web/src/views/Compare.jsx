import { useState } from "react";
import * as d3 from "d3";
import { getJSON } from "../api.js";
import { Card, ChartTableToggle, ColumnChart, DataTable, Empty, GradientLegend, Matrix, Select, diverging, fmt, useAsync, useDark } from "../charts.jsx";

export default function CompareView({ run, runs }) {
  const peers = runs.filter((r) => r.name !== run.name && r.manifest.model_id === run.manifest.model_id);
  const [other, setOther] = useState(peers[0]?.name ?? "");
  if (!peers.length)
    return (
      <Empty command={`moescope profile --model ${run.manifest.model_id} --prompts other.jsonl --out runs/other`}>
        Compare needs a second run of {run.manifest.model_id} on a different workload.
      </Empty>
    );
  return <Body run={run} peers={peers} other={other || peers[0].name} setOther={setOther} />;
}

function Body({ run, peers, other, setOther }) {
  const dark = useDark();
  const [layer, setLayer] = useState(null);
  const [mode, setMode] = useState("chart");
  const res = useAsync(() => getJSON(`/api/compare?a=${encodeURIComponent(run.name)}&b=${encodeURIComponent(other)}`), [run.name, other]);
  const picker = <Select label="Compare with" value={other} onChange={setOther} testid="compare-select" options={peers.map((p) => ({ value: p.name, label: p.name }))} />;

  if (res.loading) return <Card title="Workload compare" actions={picker}><div className="text-[13px] text-ink-2">Comparing…</div></Card>;
  if (res.error) return <Card title="Workload compare" actions={picker}><Empty>{res.error.message}</Empty></Card>;
  const c = res.data;
  const delta = c.share_b.map((row, l) => row.map((v, e) => v - c.share_a[l][e]));
  const extent = d3.max(delta.flat().map(Math.abs)) || 0.01;
  const color = diverging(extent, dark);
  const focus = layer ?? c.layers.reduce((best, l) => (l.js_distance > best.js_distance ? l : best)).layer;
  const fl = c.layers[focus];

  return (
    <div className="flex flex-col gap-4">
      <Card
        testid="view-compare"
        title={`${c.a} vs ${c.b}`}
        subtitle={`Jensen-Shannon distance between the two workloads' expert shares, per layer: 0 means the same experts in the same proportions, 1 means no overlap. Mean ${fmt.dec(c.mean_js_distance)}.`}
        actions={<>{picker}<ChartTableToggle value={mode} onChange={setMode} /></>}
      >
        {mode === "chart" ? (
          <ColumnChart
            testid="compare-js"
            data={c.layers.map((l) => ({ x: l.layer, y: l.js_distance }))}
            yFormat={fmt.dec2}
            xLabel="MoE layer"
            yLabel="JS distance"
            selected={focus}
            onSelect={setLayer}
            tooltip={(d) => `Layer ${d.x}: JS distance ${fmt.dec(d.y)}`}
          />
        ) : (
          <DataTable
            columns={[
              { key: "layer", label: "Layer" },
              { key: "js_distance", label: "JS distance", format: fmt.dec },
              { key: "gainers", label: `More in ${c.b}`, align: "left", format: (v) => v.map((g) => `${g.expert} (+${fmt.pct(g.delta)})`).join(", ") },
              { key: "losers", label: `Less in ${c.b}`, align: "left", format: (v) => v.map((g) => `${g.expert} (${fmt.pct(g.delta)})`).join(", ") },
            ]}
            rows={c.layers}
          />
        )}
      </Card>
      <Card
        title="Where share moves"
        subtitle={`Change in each expert's share of routed tokens going from ${c.a} to ${c.b}. Red experts get more of the ${c.b} workload, blue ones less. Click a row to inspect a layer.`}
      >
        <div className="mb-2"><GradientLegend color={color} min={-extent} max={extent} format={(v) => `${v > 0 ? "+" : ""}${fmt.pct(v)}`} label="Share change" /></div>
        <Matrix
          testid="compare-delta"
          values={delta}
          color={color}
          rowTitle="MoE layer"
          colTitle="Expert"
          highlightRow={focus}
          onCellClick={(r) => setLayer(r)}
          tooltip={(r, e, v) => (
            <div>
              <div className="font-medium">Layer {r} · expert {e}</div>
              <div className="tabular text-ink-2">{fmt.pct(c.share_a[r][e])} → {fmt.pct(c.share_b[r][e])} ({v > 0 ? "+" : ""}{fmt.pct(v)})</div>
            </div>
          )}
        />
        <div className="mt-3 grid gap-3 text-[13px] sm:grid-cols-2">
          <div><span className="font-medium">Layer {focus}: more in {c.b}</span><div className="text-ink-2">{fl.gainers.map((g) => `expert ${g.expert} (+${fmt.pct(g.delta)})`).join(", ") || "–"}</div></div>
          <div><span className="font-medium">Layer {focus}: less in {c.b}</span><div className="text-ink-2">{fl.losers.map((g) => `expert ${g.expert} (${fmt.pct(g.delta)})`).join(", ") || "–"}</div></div>
        </div>
      </Card>
    </div>
  );
}
