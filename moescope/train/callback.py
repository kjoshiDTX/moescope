"""Hugging Face TrainerCallback that watches MoE routing during fine-tuning.

    trainer = Trainer(..., callbacks=[MoEWatchCallback(log_dir="runs/ft")])

Every `log_every` optimizer steps it writes one line to <log_dir>/train.jsonl:
  aux_loss        Switch-style load-balancing loss, E * sum_i(f_i * P_i), averaged over layers
                  (f_i = share of routed assignments, P_i = mean router probability; 1.0 = balanced)
  z_loss          mean squared log-sum-exp of the router logits
  logit_max_mean  mean of each token's largest router logit
  load            per layer, each expert's share of routed assignments
  max_load_ratio  per layer, the busiest expert's share divided by a fair share (1/E)
  drop_rate       capacity drop rate where the model exposes one, else null
  collapsed_layers layers where one expert has held > collapse_factor x its fair share for
                  collapse_steps steps
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from transformers import TrainerCallback

from ..adapters.registry import adapter_for

logger = logging.getLogger("moescope")


class CollapseDetector:
    """Flags a layer once its max load ratio stays above `factor` for `steps` training steps."""

    def __init__(self, factor: float = 3.0, steps: int = 200):
        self.factor = factor
        self.steps = steps
        self.since: dict[int, int] = {}
        self.warned: set[int] = set()

    def update(self, step: int, ratios: list[float]) -> list[int]:
        collapsed = []
        for layer, ratio in enumerate(ratios):
            if ratio > self.factor:
                start = self.since.setdefault(layer, step)
                if step - start >= self.steps:
                    collapsed.append(layer)
            else:
                self.since.pop(layer, None)
                self.warned.discard(layer)
        return collapsed

    def new_warnings(self, collapsed: list[int]) -> list[int]:
        fresh = [l for l in collapsed if l not in self.warned]
        self.warned.update(fresh)
        return fresh


class MoEWatchCallback(TrainerCallback):
    def __init__(self, log_dir: str | Path, log_every: int = 10, collapse_factor: float = 3.0, collapse_steps: int = 200):
        self.log_dir = Path(log_dir)
        self.log_every = log_every
        self.detector = CollapseDetector(collapse_factor, collapse_steps)
        self._handles: list = []
        self._active = False
        self._reset()

    # ---- hooks ------------------------------------------------------------------------------
    def _reset(self) -> None:
        self._counts: dict[int, torch.Tensor] = {}
        self._probs: dict[int, torch.Tensor] = {}
        self._tokens: dict[int, int] = {}
        self._z = self._lmax = None
        self._z_n = 0

    def _hook(self, layer: int):
        def hook(module, inputs, out):
            if not (self._active and module.training):
                return
            with torch.no_grad():
                logits, indices = self.adapter.unpack(out)
                logits = logits.detach().float().reshape(-1, logits.shape[-1])
                E = logits.shape[-1]
                counts = torch.bincount(indices.reshape(-1), minlength=E).float()
                probs = torch.softmax(logits, dim=-1).sum(0)
                lse = torch.logsumexp(logits, dim=-1)
                if layer in self._counts:
                    self._counts[layer] += counts
                    self._probs[layer] += probs
                    self._tokens[layer] += logits.shape[0]
                else:
                    self._counts[layer], self._probs[layer], self._tokens[layer] = counts, probs, logits.shape[0]
                z, lmax = (lse ** 2).sum(), logits.max(-1).values.sum()
                self._z = z if self._z is None else self._z + z
                self._lmax = lmax if self._lmax is None else self._lmax + lmax
                self._z_n += logits.shape[0]

        return hook

    # ---- trainer events ---------------------------------------------------------------------
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        model = getattr(model, "module", model)  # unwrap DDP / accelerate wrappers
        self.adapter = adapter_for(model.config)
        routers = self.adapter.find_routers(model)
        self._handles = [m.register_forward_hook(self._hook(i)) for i, (_, m) in enumerate(routers)]
        self.num_layers = len(routers)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "train_meta.json").write_text(json.dumps({
            "model_id": getattr(model.config, "_name_or_path", "") or type(model).__name__,
            "num_layers": self.num_layers,
            "num_experts": self.adapter.num_experts,
            "top_k": self.adapter.top_k,
            "log_every": self.log_every,
            "collapse_factor": self.detector.factor,
            "collapse_steps": self.detector.steps,
        }, indent=2))
        self._log = open(self.log_dir / "train.jsonl", "a")

    def on_step_begin(self, args, state, control, **kwargs):
        self._active = (state.global_step + 1) % self.log_every == 0

    def on_step_end(self, args, state, control, **kwargs):
        if not self._active or not self._counts:
            self._active = False
            return
        self._active = False
        self.write_step(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        for h in self._handles:
            h.remove()
        self._handles = []
        if getattr(self, "_log", None):
            self._log.close()

    # ---- metrics ------------------------------------------------------------------------------
    def write_step(self, step: int) -> dict:
        E = self.adapter.num_experts
        layers = sorted(self._counts)
        load = torch.stack([self._counts[l] / self._counts[l].sum() for l in layers])
        mean_prob = torch.stack([self._probs[l] / self._tokens[l] for l in layers])
        aux = (E * (load * mean_prob).sum(-1)).mean()
        record = {
            "step": step,
            "aux_loss": float(aux),
            "z_loss": float(self._z / self._z_n),
            "logit_max_mean": float(self._lmax / self._z_n),
            "load": load.cpu().numpy().round(5).tolist(),
            "max_load_ratio": (load.max(-1).values * E).cpu().tolist(),
            "drop_rate": None,  # HF MoE layers route without a capacity limit, so nothing is dropped
            "collapsed_layers": [],
        }
        collapsed = self.detector.update(step, record["max_load_ratio"])
        record["collapsed_layers"] = collapsed
        for layer in self.detector.new_warnings(collapsed):
            logger.warning(
                "moescope: router collapse in MoE layer %d at step %d: one expert has held more than %.1fx its fair "
                "share for %d+ steps", layer, step, self.detector.factor, self.detector.steps)
        self._log.write(json.dumps(record) + "\n")
        self._log.flush()
        self._reset()
        return record
