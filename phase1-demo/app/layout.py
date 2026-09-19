"""
Deterministic conversion from an apartment constraint graph into a basic
2D floor-plan layout.

This is the first spatial stage after graph.py:
    ApartmentState -> Constraint Graph -> Room Layout

The layout engine is intentionally simple in Phase 1:
- Every room receives a rectangular footprint.
- Room dimensions are derived from the requested size.
- Rooms are placed deterministically.
- Required graph connections influence placement.
- No LLM is involved.
- The output can later be validated and optimized.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .schema import ApartmentState


@dataclass
class RoomPlacement:
    """A rectangular placement for a single room."""

    id: str
    type: str
    x: float
    y: float
    width: float
    height: float


def build_layout(state: ApartmentState, graph: nx.Graph) -> list[RoomPlacement]:
    """
    Deterministically place every room in the apartment.

    Connected rooms are placed relative to already placed neighbors.
    Disconnected rooms are placed separately below the main layout.
    """

    if not state.rooms:
        return []

    rooms_by_id = {room.id: room for room in state.rooms}
    placements: dict[str, RoomPlacement] = {}

    # ---------------------------------------------------------
    # Helper: create a placement for a room
    # ---------------------------------------------------------
    def create_placement(room, x, y):
        width, height = _room_dimensions(room.size)

        return RoomPlacement(
            id=room.id,
            type=room.type,
            x=x,
            y=y,
            width=width,
            height=height,
        )

    # ---------------------------------------------------------
    # Place each connected component
    # ---------------------------------------------------------
    component_index = 0

    for component in nx.connected_components(graph):
        component = list(component)

        # Pick the first room in this component as its anchor.
        root_id = component[0]

        if root_id in placements:
            continue

        # Separate connected components vertically.
        root_x = 0.0
        root_y = component_index * 6.0

        placements[root_id] = create_placement(
            rooms_by_id[root_id],
            root_x,
            root_y,
        )

        queue = [root_id]
        visited = {root_id}

        while queue:
            current_id = queue.pop(0)
            current_placement = placements[current_id]

            neighbors = list(graph.neighbors(current_id))

            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue

                neighbor_room = rooms_by_id[neighbor_id]

                # Place the neighbor to the right of the current room.
                neighbor_x = (
                    current_placement.x
                    + current_placement.width
                )

                neighbor_y = current_placement.y

                placements[neighbor_id] = create_placement(
                    neighbor_room,
                    neighbor_x,
                    neighbor_y,
                )

                visited.add(neighbor_id)
                queue.append(neighbor_id)

        component_index += 1

    # ---------------------------------------------------------
    # Safety: place any rooms that aren't represented in graph
    # ---------------------------------------------------------
    for room in state.rooms:
        if room.id not in placements:
            width, height = _room_dimensions(room.size)

            placements[room.id] = RoomPlacement(
                id=room.id,
                type=room.type,
                x=0.0,
                y=component_index * 6.0,
                width=width,
                height=height,
            )

            component_index += 1

    return list(placements.values())
def _room_dimensions(size: str | None) -> tuple[float, float]:
    """
    Convert the requested room size into basic rectangular dimensions.

    These values are placeholders for the first spatial prototype.
    They can later be replaced with a more principled area/aspect-ratio model.
    """

    normalized = (size or "").strip().lower()

    if normalized == "small":
        return 3.0, 3.0

    if normalized == "medium":
        return 4.0, 4.0

    if normalized == "large":
        return 5.0, 4.0

    # Safe default when the LLM does not provide a recognized size.
    return 4.0, 4.0