from shapely.geometry import Polygon
import numpy as np

def force_orthogonal_polygon(poly: Polygon) -> Polygon:
    if poly is None or poly.is_empty:
        return poly
    coords = list(poly.exterior.coords)
    if len(coords) < 3:
        return poly
    
    # Remove the last point since it's the same as the first
    coords = coords[:-1]
    
    new_coords = []
    # Start with the first point
    new_coords.append(list(coords[0]))
    
    for i in range(1, len(coords)):
        prev = new_coords[-1]
        curr = list(coords[i])
        # Find which axis to snap
        dx = abs(curr[0] - prev[0])
        dy = abs(curr[1] - prev[1])
        
        # We must alternate. But wait, if we alternate X and Y, we don't know the starting axis.
        # So we just snap the smaller difference to 0.
        if dx < dy:
            curr[0] = prev[0]  # Vertical line
        else:
            curr[1] = prev[1]  # Horizontal line
            
        new_coords.append(curr)
        
    # To close it orthogonally, the last point must connect to the first point.
    # The last edge and the first edge must be orthogonal.
    # If they are not, we might need to add a corner.
    last = new_coords[-1]
    first = new_coords[0]
    if last[0] != first[0] and last[1] != first[1]:
        # Add an intermediate point to close orthogonally
        # Choose the path that is shortest
        if abs(last[0] - first[0]) < abs(last[1] - first[1]):
            new_coords.append([first[0], last[1]])
        else:
            new_coords.append([last[0], first[1]])
            
    return Polygon(new_coords)

p = Polygon([(0.0, 0.0), (10.1, 0.2), (9.8, 10.3), (-0.2, 9.9)])
print("Before:", p.wkt)
p_ortho = force_orthogonal_polygon(p)
print("After:", p_ortho.wkt)
