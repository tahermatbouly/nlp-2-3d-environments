import pickle
import json
from pathlib import Path

def main():
    data_path = "ResPlan.pkl"
    out_dir = Path("training-data")
    out_dir.mkdir(exist_ok=True)
    
    print(f"Loading {data_path}...")
    with open(data_path, "rb") as f:
        data = pickle.load(f)
        
    print(f"Processing {len(data)} records...")
    for i, record in enumerate(data):
        record_id = record["id"]
        graph = record["graph"]
        
        rooms = []
        room_counts = {
            "bedroom": 0, "bathroom": 0, "kitchen": 0,
            "living": 0, "balcony": 0, "front_door": 0
        }
        
        for n, node_data in graph.nodes(data=True):
            rtype = node_data["type"]
            room_counts[rtype] = room_counts.get(rtype, 0) + 1
            
            connections = []
            for neighbor in graph.neighbors(n):
                edge_data = graph.get_edge_data(n, neighbor)
                connections.append({
                    "target_id": neighbor,
                    "type": edge_data["type"]
                })
                
            rooms.append({
                "id": n,
                "type": rtype,
                "area": float(node_data["area"]),
                "connections": connections
            })
            
        other_features_present = {}
        # A feature is present if its Shapely geometry is not empty
        for feature in ["wall", "window", "door", "inner", "garden", "parking", "pool", "land"]:
            geom = record.get(feature)
            other_features_present[feature] = False if geom is None else not geom.is_empty
            
        out_record = {
            "id": record_id,
            "net_area": float(record["net_area"]),
            "gross_area": float(record["area"]),
            "wall_depth": float(record["wall_depth"]),
            "rooms": rooms,
            "room_counts": room_counts,
            "other_features_present": other_features_present
        }
        
        out_file = out_dir / f"{record_id}.json"
        with open(out_file, "w") as f:
            json.dump(out_record, f, indent=2)
            
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(data)} records...")

    print(f"Done! All files saved to {out_dir}/")

if __name__ == "__main__":
    main()
