import torch
import numpy as np

cache = torch.load("cache/dataset.pt", weights_only=False)
train = cache["train"]

total_rooms = 0
v_counts = {}

for s in train:
    for i in range(len(s["nverts"])):
        V = int(s["nverts"][i])
        if V == 0: continue
        v_counts[V] = v_counts.get(V, 0) + 1
        total_rooms += 1

print(v_counts)
