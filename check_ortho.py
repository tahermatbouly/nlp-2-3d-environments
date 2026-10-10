import torch
import numpy as np

cache = torch.load("cache/dataset.pt", weights_only=False)
train = cache["train"]

total_edges = 0
ortho_edges = 0

for s in train[:100]:
    coords = s["coords"]
    for i in range(len(coords)):
        V = s["nverts"][i]
        if V == 0: continue
        c = coords[i][:V]
        for j in range(V):
            p1 = c[j]
            p2 = c[(j+1)%V]
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            if dx < 1e-3 or dy < 1e-3:
                ortho_edges += 1
            total_edges += 1

print(f"Ortho edges: {ortho_edges} / {total_edges} ( {ortho_edges/total_edges*100:.1f}% )")
