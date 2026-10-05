import pickle

with open("ResPlan.pkl", "rb") as f:
    data = pickle.load(f)

print(f"Number of plans: {len(data)}")
sample = data[0]
print("Keys in a sample plan:")
for k, v in sample.items():
    print(f"  {k}: {type(v)}")

import json
with open("split.json") as f:
    splits = json.load(f)
print("\nSplits:")
for k, v in splits.items():
    print(f"  {k}: {len(v)}")
