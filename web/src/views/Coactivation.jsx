import { useState } from "react";
import * as d3 from "d3";
import { getJSON, runPath } from "../api.js";
import { Card, ChartTableToggle, DataTable, Empty, GradientLegend, Matrix, Segmented, Select, diverging, fmt, sequential, useAsync, useDark } from "../charts.jsx";

export default function CoactivationView({ run }) {
  const L = run.manifest.num_layers;
  const [layer, setLayer] = useState("0");
  const [measure, setMeasure] = useState("lift");
  const [mode, setMode] = useState("chart");
  const dark = useDark();
  const data = useAsync(() => getJSON(`${runPath(run.name)}/coactivation?layer=${layer}`), [run.name, layer]);

  let body;
  if (data.loading) body = <div className="text-[13px] text-ink-2">Computing…</div>;
  else if (data.error) body = <Empty>Could not compute co-activation: {data.error.message}</Empty>;
  else {
    const { counts, lift, tokens } = data.data;
    const E = counts.length;
    const rate = counts.map((row) => row.map((v) => v / Math.max(tokens, 1)));
    // Pairs that never fire together (and the diagonal) draw as empty cells, not as extreme lift.
    const offDiag = (m) => m.map((row, i) => row.map((v, j) => (i === j || counts[i][j] === 0 ? null : v)));
    const log2lift = lift.map((row) => row.map((v) => (v > 0 ? Math.log2(v) : -3)));
    const values = offDiag(measure === "lift" ? log2lift : rate);
    const maxRate = d3.max(values.flat().filter((v) => v != null)) || 1;
    const color = measure === "lift" ? diverging(2, dark) : sequential(maxRate, dark);

    const pairs = [];
    for (let i = 0; i < E; i++)
      for (let j = i + 1; j < E; j++)
        if (counts[i][j] >= Math.max(5, tokens * 0.005)) pairs.push({ a: i, b: j, rate: rate[i][j], lift: lift[i][j], count: counts[i][j] });
    pairs.sort((x, y) => y.lift - x.lift);

    body = mode === "chart" ? (
      <div className="flex flex-col gap-5 xl:flex-row">
        <div className="min-w-0 flex-1">
          <div className="mb-2">
            {measure === "lift" ? (
              <GradientLegend color={color} min={-2} max={2} format={(v) => `${(2 ** v).toFixed(2)}×`} label="Lift (vs independent)" />
            ) : (
              <GradientLegend color={color} max={maxRate} format={fmt.pct} label="Tokens with both" />
            )}
          </div>
          <Matrix
            testid="coactivation-matrix"
            values={values}
            color={color}
            maxCell={18}
            rowTitle="Expert"
            colTitle="Expert"
            tooltip={(r, c) =>
              r === c ? `Expert ${r}: chosen for ${fmt.pct(rate[r][r])} of tokens` : counts[r][c] === 0 ? `Experts ${r} & ${c}: never chosen together` : (
                <div>
                  <div className="font-medium">Experts {r} & {c}</div>
                  <div className="tabular text-ink-2">together on {fmt.pct(rate[r][c])} of tokens</div>
                  <div className="tabular text-ink-2">{lift[r][c].toFixed(2)}× what independent picks would give</div>
                </div>
              )
            }
          />
        </div>
        <div className="xl:w-80">
          <h3 className="mb-2 text-[13px] font-medium text-ink">Strongest pairs</h3>
          <DataTable
            maxHeight={360}
            columns={[
              { key: "a", label: "Pair", align: "left", format: (_, r) => `${r.a} & ${r.b}` },
              { key: "rate", label: "Tokens", format: fmt.pct },
              { key: "lift", label: "Lift", format: (v) => `${v.toFixed(2)}×` },
            ]}
            rows={pairs.slice(0, 12)}
          />
        </div>
      </div>
    ) : (
      <DataTable
        columns={[
          { key: "a", label: "Expert A" },
          { key: "b", label: "Expert B" },
          { key: "count", label: "Tokens", format: fmt.num },
          { key: "rate", label: "Share", format: fmt.pct },
          { key: "lift", label: "Lift", format: (v) => `${v.toFixed(2)}×` },
        ]}
        rows={pairs}
      />
    );
  }

  return (
    <Card
      testid="view-coactivation"
      title={`Co-activation · layer ${layer}`}
      subtitle="How often two experts land in the same token's top-k. Lift compares that with what you'd expect if each expert were picked independently at its own rate: red pairs fire together more than chance, blue ones less; empty cells never fire together."
      actions={
        <>
          <Select label="Layer" value={layer} onChange={setLayer} testid="layer-select"
            options={d3.range(L).map((l) => ({ value: String(l), label: String(l) }))} />
          <Segmented value={measure} onChange={setMeasure} options={[{ value: "lift", label: "Lift" }, { value: "rate", label: "Pair rate" }]} />
          <ChartTableToggle value={mode} onChange={setMode} />
        </>
      }
    >
      {body}
    </Card>
  );
}
