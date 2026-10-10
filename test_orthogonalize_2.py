from shapely.geometry import Polygon
import numpy as np

def force_orthogonal_polygon(poly: Polygon) -> Polygon:
    if poly is None or poly.is_empty:
        return poly
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return poly
    
    coords = coords[:-1]
    new_coords = [list(coords[0])]
    
    for i in range(1, len(coords)):
        prev = new_coords[-1]
        curr = list(coords[i])
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        if dx < dy:
            curr[0] = prev[0]
        else:
            curr[1] = prev[1]
        
        # avoid duplicate adjacent points
        if curr != prev:
            new_coords.append(curr)
            
    last = new_coords[-1]
    first = new_coords[0]
    if last[0] != first[0] and last[1] != first[1]:
        if abs(last[0] - first[0]) < abs(last[1] - first[1]):
            new_coords.append([first[0], last[1]])
        else:
            new_coords.append([last[0], first[1]])
            
    return Polygon(new_coords)

p = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
p_ortho = force_orthogonal_polygon(p)
print("After:", p_ortho.wkt)
