import argparse
import random
import pickle
import matplotlib.pyplot as plt

from floorplan_diffusion.diffusion import load_checkpoint
from floorplan_diffusion.data import load_cache
from floorplan_diffusion.generate import generate_floorplans, sample_to_graph, to_plan_dict
from resplan_utils import plot_plan_and_graph

def main():
    parser = argparse.ArgumentParser(description="Generate and visualize a floorplan from a trained model.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--data", type=str, default="ResPlan.pkl", help="Path to ResPlan.pkl")
    parser.add_argument("--split", type=str, default="split.json", help="Path to split.json")
    parser.add_argument("--cache", type=str, default="cache", help="Dataset cache directory")
    parser.add_argument("--out", type=str, default="inference_out.png", help="Output image path")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda/cpu)")
    parser.add_argument("--steps", type=int, default=50, help="DDPM sampling steps")
    parser.add_argument("--samples", type=int, default=8, help="Number of samples to draw (best is chosen)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--idx", type=int, default=-1, help="Specific test index to use (default: random)")
    args = parser.parse_args()

    print("Loading dataset cache...")
    cache = load_cache(args.data, args.split, args.cache)
    test_split = cache["test"]
    
    idx = args.idx if args.idx >= 0 else random.randint(0, len(test_split) - 1)
    sample = test_split[idx]
    
    # Reconstruct the conditioning graph (nodes/edges/areas, no geometry)
    G = sample_to_graph(sample)
    
    print(f"Loading checkpoint from {args.ckpt}...")
    model = load_checkpoint(args.ckpt, device=args.device)
    
    print(f"Generating {args.samples} samples for test plan ID {sample['id']}...")
    # generate_floorplans automatically ranks by topological/geometric validity
    cands = generate_floorplans(model, [G], K=args.samples, steps=args.steps, 
                                device=args.device, seed=args.seed)[0]
    
    top = cands[0]
    print(f"Top candidate valid: {top.valid}, score: {top.score:.3f}")
    
    # Convert back to ResPlan polygon format
    plan_dict = to_plan_dict(top, plan_id=sample['id'])
    
    print("Plotting results...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Load original dataset to get the Ground Truth geometry for comparison
    with open(args.data, "rb") as f:
        all_plans = pickle.load(f)
    gt_plan = next(p for p in all_plans if p["id"] == sample["id"])
    
    plot_plan_and_graph(gt_plan, ax=axes[0], title=f"Ground Truth (ID {sample['id']})")
    plot_plan_and_graph(plan_dict, ax=axes[1], title=f"Generated (Score: {top.score:.2f})")
    
    plt.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"Saved visualization to {args.out}")

if __name__ == "__main__":
    main()
