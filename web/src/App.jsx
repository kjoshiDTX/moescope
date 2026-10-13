import { useEffect, useState } from "react";
import { getJSON } from "./api.js";
import { Empty, fmt, useAsync } from "./charts.jsx";
import UtilizationView from "./views/Utilization.jsx";
import CoverageView from "./views/Coverage.jsx";
import CoactivationView from "./views/Coactivation.jsx";
import CompareView from "./views/Compare.jsx";
import CacheView from "./views/Cache.jsx";
import PruneView from "./views/Prune.jsx";
import TrainingView from "./views/Training.jsx";

const VIEWS = [
  { id: "utilization", label: "Utilization", component: UtilizationView },
  { id: "coverage", label: "Coverage", component: CoverageView },
  { id: "coactivation", label: "Co-activation", component: CoactivationView },
  { id: "compare", label: "Compare", component: CompareView },
  { id: "cache", label: "Cache sim", component: CacheView },
  { id: "prune", label: "Prune", component: PruneView },
  { id: "training", label: "Training", component: TrainingView },
];

function readHash() {
  const params = new URLSearchParams(window.location.hash.slice(1));
  return { view: params.get("view") ?? "utilization", run: params.get("run") };
}

function ThemeToggle() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem("moescope-theme") ?? "system"; } catch { return "system"; }
  });
  useEffect(() => {
    if (theme === "system") delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = theme;
    try { localStorage.setItem("moescope-theme", theme); } catch { /* storage unavailable */ }
  }, [theme]);
  const next = { system: "light", light: "dark", dark: "system" }[theme];
  return (
    <button onClick={() => setTheme(next)} className="rounded-md border border-line px-2 py-1 text-[12px] text-ink-2 hover:text-ink" title="Switch theme">
      Theme: {theme}
    </button>
  );
}

export default function App() {
  const [{ view, run }, setNav] = useState(readHash);
  const runs = useAsync(() => getJSON("/api/runs"), []);

  useEffect(() => {
    const onHash = () => setNav(readHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const go = (next) => {
    const merged = { view, run, ...next };
    const params = new URLSearchParams();
    params.set("view", merged.view);
    if (merged.run) params.set("run", merged.run);
    window.location.hash = params.toString();
  };

  const runList = runs.data ?? [];
  const current = runList.find((r) => r.name === run) ?? runList[0];
  const View = (VIEWS.find((v) => v.id === view) ?? VIEWS[0]).component;

  return (
    <div className="min-h-screen bg-page text-ink">
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-3 sm:px-6">
        <div className="flex items-baseline gap-3">
          <span className="text-[16px] font-semibold">moescope</span>
          <span className="hidden text-[13px] text-ink-2 sm:inline">which experts your workload actually uses</span>
        </div>
        <ThemeToggle />
      </header>
      <div className="flex flex-col lg:flex-row">
        <aside className="border-b border-line px-4 py-3 lg:w-64 lg:shrink-0 lg:border-b-0 lg:border-r lg:px-3 lg:py-4">
          <div className="mb-2 px-1 text-[12px] font-medium text-muted">Runs</div>
          {runs.loading && <div className="px-1 text-[13px] text-ink-2">Loading…</div>}
          {runs.error && <div className="px-1 text-[13px] text-critical">Could not load runs: {runs.error.message}</div>}
          <nav className="flex gap-1 overflow-x-auto lg:flex-col" data-testid="run-list">
            {runList.map((r) => (
              <button
                key={r.name}
                onClick={() => go({ run: r.name })}
                className={`shrink-0 rounded-lg px-2.5 py-2 text-left ${current?.name === r.name ? "bg-wash" : "hover:bg-wash"}`}
              >
                <div className="text-[13px] font-medium text-ink">{r.name}</div>
                <div className="text-[12px] text-muted">
                  {r.manifest.model_id.split("/").at(-1)}{r.tokens != null && ` · ${fmt.num(r.tokens)} tok`}
                </div>
              </button>
            ))}
          </nav>
        </aside>
        <main className="min-w-0 flex-1 px-4 py-4 sm:px-6">
          <nav className="mb-4 flex gap-1 overflow-x-auto border-b border-line" data-testid="tabs">
            {VIEWS.map((v) => (
              <button
                key={v.id}
                data-testid={`tab-${v.id}`}
                onClick={() => go({ view: v.id })}
                className={`-mb-px shrink-0 border-b-2 px-3 py-2 text-[13px] ${view === v.id ? "border-accent font-medium text-ink" : "border-transparent text-ink-2 hover:text-ink"}`}
              >
                {v.label}
              </button>
            ))}
          </nav>
          {runs.data && runList.length === 0 && view !== "training" ? (
            <Empty command="moescope profile --model … --prompts … --out runs/my-run">
              No runs in this folder yet. Profile a model first.
            </Empty>
          ) : (
            (current || view === "training") && <View run={current} runs={runList} key={`${view}-${current?.name}`} />
          )}
        </main>
      </div>
    </div>
  );
}
