import { useState } from "react";
import * as d3 from "d3";
import { getJSON } from "../api.js";
import { Card, ChartTableToggle, DataTable, Empty, GradientLegend, LineChart, Matrix, Select, fmt, sequential, useAsync, useDark } from "../charts.jsx";

export default function TrainingView() {
  const logs = useAsync(() => getJSON("/api/train"), []);
  if (logs.loading) return <div className="text-[13px] text-ink-2">Loading training logs…</div>;
  if (!logs.data?.length)
    return (
      <Empty command='Trainer(..., callbacks=[MoEWatchCallback(log_dir="runs/ft")])'>
        No training logs found. Add the callback to a Hugging Face Trainer.
      </Empty>
    );
  return <Body logs={logs.data} />;
}

function Body({ logs }) {
  const dark = useDark();
  const [which, setWhich] = useState(logs[0].name);
  const [mode, setMode] = useState("chart");
  const log = logs.find((l) => l.name === which) ?? logs[0];
  const { steps, meta } = log;
  if (!steps.length) return <Empty>This training log has no steps yet.</Empty>;
  const factor = meta.collapse_factor ?? 3;
  const line = (field, label, color = "var(--series-1)") => [{ key: field, label, color, points: steps.filter((s) => s[field] != null).map((s) => ({ x: s.step, y: s[field] })) }];
  const maxRatio = steps.map((s) => ({ x: s.step, y: d3.max(s.max_load_ratio) }));
  const ratioByLayer = d3.transpose(steps.map((s) => s.max_load_ratio)); // layers x steps
  const ratioColor = sequential(Math.max(factor * 1.5, d3.max(ratioByLayer.flat())), dark);
  const last = steps.at(-1);
  const loadColor = sequential(d3.max(last.load.flat()), dark);
  const collapsed = [...new Set(steps.flatMap((s) => s.collapsed_layers))].sort((a, b) => a - b);
  const firstFlag = (layer) => steps.find((s) => s.collapsed_layers.includes(layer))?.step;
  const hasDrops = steps.some((s) => s.drop_rate != null);

  return (
    <div className="flex flex-col gap-4">
      {collapsed.length > 0 && (
        <div data-testid="collapse-warning" role="alert" className="flex gap-2 rounded-xl border px-4 py-3 text-[14px]" style={{ borderColor: "var(--critical)" }}>
          <span aria-hidden className="font-bold" style={{ color: "var(--critical)" }}>!</span>
          <div>
            <span className="font-semibold">Router collapse warning.</span>{" "}
            {collapsed.map((l) => `layer ${l} (from step ${firstFlag(l)})`).join(", ")}: one expert has held more than {factor}× its fair share for {meta.collapse_steps ?? 200}+ steps.
          </div>
        </div>
      )}
      <Card
        testid="view-training"
        title="Router load balance"
        subtitle={`Largest expert's load relative to an even split (1× = balanced), worst layer at each logged step. Above ${factor}× for ${meta.collapse_steps ?? 200} steps counts as collapse.`}
        actions={<>{logs.length > 1 && <Select label="Log" value={which} onChange={setWhich} options={logs.map((l) => ({ value: l.name, label: l.name }))} />}<ChartTableToggle value={mode} onChange={setMode} /></>}
      >
        {mode === "chart" ? (
          <LineChart testid="train-ratio" series={[{ key: "r", label: "Max load ratio", color: "var(--series-1)", points: maxRatio }]} xLabel="Step" yFormat={(v) => `${v.toFixed(1)}×`} refLines={[{ y: factor, label: `${factor}×` }]} height={220} markers={false} />
        ) : (
          <DataTable
            columns={[
              { key: "step", label: "Step" },
              { key: "aux_loss", label: "Aux loss", format: (v) => (v == null ? "–" : v.toFixed(4)) },
              { key: "z_loss", label: "Z-loss", format: (v) => (v == null ? "–" : v.toFixed(4)) },
              { key: "max_load_ratio", label: "Max load ratio", format: (v) => `${d3.max(v).toFixed(2)}×` },
              { key: "drop_rate", label: "Drop rate", format: (v) => (v == null ? "n/a" : fmt.pct(v)) },
              { key: "collapsed_layers", label: "Collapsed layers", align: "left", format: (v) => v.join(", ") || "–" },
            ]}
            rows={steps}
          />
        )}
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Aux load-balancing loss" subtitle="Switch-style: num_experts × Σ (token fraction × mean router prob). 1.0 is perfectly balanced.">
          <LineChart testid="train-aux" series={line("aux_loss", "Aux loss")} xLabel="Step" yFormat={(v) => v.toFixed(2)} height={200} markers={false} />
        </Card>
        <Card title="Router z-loss" subtitle="Mean squared log-sum-exp of router logits. Rising z-loss means logits are growing.">
          <LineChart testid="train-z" series={line("z_loss", "Z-loss", "var(--series-1)")} xLabel="Step" yFormat={(v) => v.toFixed(2)} height={200} markers={false} />
        </Card>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Load ratio by layer over training" subtitle="Each row is a layer, each column a logged step.">
          <div className="mb-2"><GradientLegend color={ratioColor} min={1} max={Math.max(factor * 1.5, d3.max(ratioByLayer.flat()))} format={(v) => `${v.toFixed(1)}×`} /></div>
          <Matrix values={ratioByLayer} color={ratioColor} rowTitle="Layer" colTitle="Logged step" colLabel={(c) => steps[c].step} maxCell={14}
            tooltip={(r, c, v) => `Layer ${r} at step ${steps[c].step}: ${v.toFixed(2)}× fair share`} />
        </Card>
        <Card title={`Per-expert load at step ${last.step}`} subtitle={hasDrops ? `Capacity drop rate: ${fmt.pct(last.drop_rate)}` : "This model does not expose capacity drops."}>
          <div className="mb-2"><GradientLegend color={loadColor} max={d3.max(last.load.flat())} format={fmt.pct} label="Token share" /></div>
          <Matrix values={last.load} color={loadColor} rowTitle="Layer" colTitle="Expert" maxCell={14}
            tooltip={(r, c, v) => `Layer ${r} · expert ${c}: ${fmt.pct(v)} of routed tokens`} />
        </Card>
      </div>
    </div>
  );
}
