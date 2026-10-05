import pickle
with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

sample = data[0]
g = sample["graph"]

print(f"Graph nodes: {g.number_of_nodes()}")
print(f"Graph edges: {g.number_of_edges()}")

print("\nNode features:")
for n, d in list(g.nodes(data=True))[:3]:
    print(f"  {n}: {list(d.keys())}")

print("\nEdge features:")
for u, v, d in list(g.edges(data=True))[:3]:
    print(f"  ({u}, {v}): {list(d.keys())}, type: {d.get('type')}")
