"""Evaluate a trained checkpoint on the held-out split and optionally plot samples.

    python -m floorplan_diffusion.evaluate --ckpt checkpoints/diffusion/best.pt --split test --k 8
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import torch

from .data import load_cache
from .diffusion import load_checkpoint
from .generate import generate_floorplans, sample_to_graph, to_plan_dict
from .metrics import evaluate_ground_truth, evaluate_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--data", default="ResPlan.pkl")
    p.add_argument("--split_file", default="split.json")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--n", type=int, default=0, help="limit #plans (0 = all)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="")
    p.add_argument("--plot", type=int, default=0, help="save this many generated-vs-GT figures")
    p.add_argument("--no_ema", action="store_true")
    a = p.parse_args()

    cache = load_cache(a.data, a.split_file, a.cache_dir)
    samples = cache[a.split][: a.n] if a.n else cache[a.split]
    model = load_checkpoint(a.ckpt, a.device, use_ema=not a.no_ema)
    print(f"[eval] {a.split}: {len(samples)} plans, K={a.k}, steps={a.steps}")
    res = evaluate_model(model, samples, K=a.k, steps=a.steps, eta=a.eta, guidance=a.guidance,
                         device=a.device, batch_size=a.batch_size)
    res["ground_truth_reference"] = evaluate_ground_truth(samples)
    print(json.dumps(res, indent=2))
    out = a.out or os.path.join(os.path.dirname(a.ckpt), f"eval_{a.split}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[eval] wrote {out}")

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from resplan_utils import plot_plan
        from .generate import build_candidate
        from .metrics import gt_polygons
        sub = samples[: a.plot]
        graphs = [sample_to_graph(s) for s in sub]
        cands = generate_floorplans(model, graphs, K=a.k, steps=a.steps, eta=a.eta,
                                    guidance=a.guidance, device=a.device)
        os.makedirs(os.path.join(os.path.dirname(out), "plots"), exist_ok=True)
        for s, G, cs in zip(sub, graphs, cands):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            gt = build_candidate(G, [np.asarray(p.exterior.coords)[:-1] for p in gt_polygons(s)])
            for ax, c, t in zip(axs, [gt, cs[0], cs[-1]],
                                ["ground truth", f"top-1 (score {cs[0].score:.2f}, valid={cs[0].valid})",
                                 f"worst of {len(cs)}"]):
                plot_plan(to_plan_dict(c, s["id"]), ax=ax, title=t, legend=False)
            fig.savefig(os.path.join(os.path.dirname(out), "plots", f"plan_{s['id']}.png"), dpi=90)
            plt.close(fig)


if __name__ == "__main__":
    main()
