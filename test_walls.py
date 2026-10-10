import pickle
from resplan_utils import plot_plan
from floorplan_diffusion.generate import sample_to_graph, to_plan_dict
import matplotlib.pyplot as plt

with open("ResPlan.pkl", "rb") as f:
    plans = pickle.load(f)

p = plans[0]

# Let's write our own procedural walls and doors to see how it looks
from shapely.ops import unary_union
from shapely.geometry import Point, MultiPolygon, Polygon
import networkx as nx

# Re-run a mocked to_plan_dict on the ground truth graph
from floorplan_diffusion.generate import Candidate
G = sample_to_graph(p)
# Put ground truth geometries into the graph
for n in G.nodes():
    room_type = G.nodes[n]["type"]
    if room_type in p:
        geom = p[room_type]
        if isinstance(geom, MultiPolygon):
            geom = geom.geoms[0] # just take first for mock
        G.nodes[n]["geometry"] = geom

cand = Candidate(graph=G, polygons=[])
plan = to_plan_dict(cand)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_plan(plan, ax=ax)
plt.savefig("test_walls_before.png")

# Now modify plan
allg = [d["geometry"] for _, d in G.nodes(data=True) if "geometry" in d and not d["geometry"].is_empty]
boundaries = unary_union([g.boundary for g in allg if g.is_valid])
wall = boundaries.buffer(1.5, cap_style=2, join_style=2)
if "door" in plan:
    wall = wall.difference(plan["door"])
plan["wall"] = wall

fig, ax = plt.subplots(1, 1, figsize=(8, 8))
plot_plan(plan, ax=ax)
plt.savefig("test_walls_after.png")
print("Done!")
