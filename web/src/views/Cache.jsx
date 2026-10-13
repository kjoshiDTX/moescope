import { useState } from "react";
import { getJSON, runPath } from "../api.js";
import { Card, ChartTableToggle, DataTable, Empty, LineChart, fmt, useAsync } from "../charts.jsx";

// Fixed slot per policy: color follows the policy, never its rank.
const POLICY = {
  lru: { label: "LRU", color: "var(--series-1)" },
  lfu: { label: "LFU", color: "var(--series-2)" },
  pinned: { label: "Pinned", color: "var(--series-3)" },
};

export default function CacheView({ run }) {
  const sim = useAsync(() => getJSON(`${runPath(run.name)}/sim`), [run.name]);
  if (sim.loading) return <div className="text-[13px] text-ink-2">Loading simulation…</div>;
  if (sim.error)
    return <Empty command={`moescope simulate ${run.name} --budget-gb 1,2,4,8`}>No cache simulation for this run yet.</Empty>;
  return <Body run={run} sim={sim.data} />;
}

function Body({ run, sim }) {
  const [mode, setMode] = useState("chart");
  const policies = [...new Set(sim.results.map((r) => r.policy))];
  const series = (field) =>
    policies.map((p) => ({
      key: p,
      label: POLICY[p]?.label ?? p,
      color: POLICY[p]?.color ?? "var(--muted)",
      points: sim.results.filter((r) => r.policy === p).map((r) => ({ x: r.budget_gb, y: r[field] })),
    }));
  const rec = sim.recommendation;
  const choice = rec.choice;
  const m = run.manifest;
  const total = m.num_layers * m.num_experts * m.expert_bytes;
  const gib = (v) => `${v} GiB`;

  return (
    <div className="flex flex-col gap-4">
      <Card
        testid="view-cache"
        title="Expert-cache hit rate by memory budget"
        subtitle={`Replays every expert lookup${sim.phase !== "all" ? ` (${sim.phase} only)` : ""} in token order. LRU and LFU start cold; Pinned preloads the most-used experts and never evicts. All ${m.num_layers * m.num_experts} routed experts take ${fmt.bytes(total)}.`}
        actions={<ChartTableToggle value={mode} onChange={setMode} />}
      >
        {mode === "chart" ? (
          <LineChart testid="cache-chart" series={series("hit_rate")} xFormat={gib} xLabel="Expert-cache budget" yLabel="Hit rate" yDomain={[0, 1]} />
        ) : (
          <DataTable
            testid="cache-table"
            columns={[
              { key: "budget_gb", label: "Budget", format: gib },
              { key: "slots", label: "Slots" },
              { key: "policy", label: "Policy", align: "left", format: (v) => POLICY[v]?.label ?? v },
              { key: "hit_rate", label: "Hit rate", format: fmt.pct },
              { key: "bytes_per_token", label: "Moved per token", format: fmt.bytes },
            ]}
            rows={sim.results}
          />
        )}
      </Card>
      <div className="grid gap-4 xl:grid-cols-2">
        <Card title="Weights moved per token" subtitle="Bytes fetched into the cache per generated or prefilled token. Lower is faster on bandwidth-bound hardware.">
          <LineChart series={series("bytes_per_token")} xFormat={gib} xLabel="Expert-cache budget" yFormat={(v) => `${(v / 2 ** 20).toFixed(0)} MiB`} height={220} />
        </Card>
        <Card testid="recommendation" title="Recommendation">
          <p className="text-[14px] text-ink">
            <span className="font-semibold">{choice.budget_gb} GiB with {POLICY[choice.policy]?.label ?? choice.policy}</span>{" "}
            gives a {fmt.pct(choice.hit_rate)} hit rate, which {choice.meets_target ? "meets" : "falls short of"} the {fmt.pct0(rec.target_hit_rate)} target.
          </p>
          <div className="mt-3 text-[13px] text-ink-2">
            {rec.llamacpp.error ? (
              <p>{rec.llamacpp.error}. Engine flags are only printed after reading them from your llama.cpp build.</p>
            ) : rec.settings.length ? (
              <>
                <p>Checked {rec.llamacpp.binary} (version {rec.llamacpp.version ?? "unknown"}):</p>
                <code className="mt-1 block rounded bg-wash px-2 py-1 font-mono text-[12px] text-ink">
                  {rec.settings.map((s) => `${s.flag}${s.value != null ? ` ${s.value}` : ""}`).join(" ")}
                </code>
                <ul className="mt-1 list-disc pl-5">{rec.settings.map((s) => <li key={s.flag}>{s.flag}: {s.why}</li>)}</ul>
              </>
            ) : (
              <p>{rec.llamacpp.binary} lists none of the MoE offload flags moescope knows.</p>
            )}
          </div>
          <h3 className="mt-4 mb-1 text-[13px] font-medium text-ink">Pin list ({sim.pins.length} experts, most-used first)</h3>
          <DataTable
            maxHeight={200}
            columns={[{ key: "layer", label: "Layer" }, { key: "expert", label: "Expert" }, { key: "lookups", label: "Lookups", format: fmt.num }]}
            rows={sim.pins.slice(0, 50)}
          />
        </Card>
      </div>
    </div>
  );
}
