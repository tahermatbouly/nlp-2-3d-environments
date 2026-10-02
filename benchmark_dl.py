import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.dataset import FloorPlanDataset
from torch_geometric.loader import DataLoader
import pipeline.config as cfg

print("Loading dataset...")
import pickle
with open(cfg.DATA_PKL, "rb") as f:
    all_plans = pickle.load(f)

train_ds = FloorPlanDataset(split="train", plans=all_plans)
train_loader = DataLoader(train_ds, batch_size=4, shuffle=False, num_workers=4)

t0 = time.time()
print("Benchmarking DataLoader...")
for i, batch in enumerate(train_loader):
    if i == 50:
        break
t1 = time.time()
print(f"Time for 50 batches: {t1 - t0:.2f} seconds ({(t1 - t0) / 50:.2f} s/batch)")
