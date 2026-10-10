import torch

cache = torch.load("cache/dataset.pt", weights_only=False)
train = cache["train"]

total_rooms = 0
alt_rooms = 0
non_alt_edges = 0

for s in train[:100]:
    coords = s["coords"]
    for i in range(len(coords)):
        V = s["nverts"][i]
        if V == 0: continue
        c = coords[i][:V]
        
        is_alt = True
        for j in range(V):
            p1 = c[j]
            p2 = c[(j+1)%V]
            dx = abs(p1[0] - p2[0])
            dy = abs(p1[1] - p2[1])
            # For strict alternation, one of dx or dy must be 0
            if dx > 1e-3 and dy > 1e-3:
                is_alt = False
                non_alt_edges += 1
        
        if is_alt:
            alt_rooms += 1
        total_rooms += 1

print(f"Alt rooms: {alt_rooms} / {total_rooms} ( {alt_rooms/total_rooms*100:.1f}% )")
