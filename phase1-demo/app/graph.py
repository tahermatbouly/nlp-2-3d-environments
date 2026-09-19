"""
Deterministic conversion from ApartmentState (the LLM-extracted JSON) into a
bubble constraint graph: nodes = rooms, edges = required adjacencies.

This is the strict Phase-1 boundary described in PHASE_1.md section 6:
the LLM's job ends at producing structured JSON. Everything from here on
is pure, deterministic Python -- no LLM involvement, fully unit-testable,
and guaranteed to produce the same graph for the same input every time.
"""

from __future__ import annotations

import networkx as nx

from .schema import ApartmentState


def build_graph(state: ApartmentState) -> nx.Graph:
    """
    Convert an ApartmentState into a NetworkX graph.

    Nodes are keyed by room `id` (guaranteed unique per the schema) and carry
    `type`, `count`, and `size` as node attributes.

    Edges represent required adjacency. Each room's `connections` list may
    reference *either* another room's exact `id` (e.g. "bedroom_1") *or* a
    bare room `type` (e.g. "bedroom", "kitchen") -- both forms appear in the
    project's own examples (PHASE_1.md uses bare types; prompts.py instructs
    the LLM to use ids), so both must resolve correctly or edges will be
    silently dropped depending on how the LLM happened to phrase a given
    response.

    Resolution order for a connection reference `ref`:
      1. If `ref` matches an existing room id exactly, use that room.
      2. Otherwise, if `ref` matches a room `type`, use the first room of
         that type (by list order). This mirrors "connect me to a bedroom"
         being satisfiable by any bedroom when the user didn't pick one.
      3. If neither matches, the reference is unresolved and is skipped
         rather than crashing the whole conversion -- but it is reported
         back to the caller (see `build_graph_report`) so the mismatch is
         visible instead of silently swallowed.

    Connections are treated as bidirectional: an edge is added regardless
    of which side declared it, and duplicate/reverse declarations collapse
    into the same single edge (a plain nx.Graph, not a MultiGraph, already
    guarantees this for us).
    """
    graph, _ = build_graph_report(state)
    return graph


def build_graph_report(state: ApartmentState) -> tuple[nx.Graph, list[str]]:
    """
    Same as `build_graph`, but also returns a list of human-readable
    warnings for any connection reference that could not be resolved.
    An empty warnings list means the conversion is unambiguous and total.
    """
    G = nx.Graph()
    warnings: list[str] = []

    # First pass: every room becomes a node, regardless of whether it has
    # any connections. This guarantees rooms mentioned only in the
    # "requirements" counts but never individually connected still show up
    # as isolated nodes rather than vanishing from the graph.
    for room in state.rooms:
        G.add_node(room.id, type=room.type, count=room.count, size=room.size)

    # Build a type -> first-matching-room-id lookup for fallback resolution.
    type_to_first_id: dict[str, str] = {}
    for room in state.rooms:
        type_to_first_id.setdefault(room.type, room.id)

    room_ids = {room.id for room in state.rooms}

    # Second pass: resolve and add edges.
    for room in state.rooms:
        for ref in room.connections:
            target_id = _resolve_connection(ref, room.id, room_ids, type_to_first_id)
            if target_id is None:
                warnings.append(
                    f"Room '{room.id}' declares a connection to '{ref}', "
                    f"which matches neither an existing room id nor a room type. Skipped."
                )
                continue
            if target_id == room.id:
                warnings.append(
                    f"Room '{room.id}' declares a connection to itself ('{ref}'). Skipped."
                )
                continue
            G.add_edge(room.id, target_id)

    return G, warnings


def _resolve_connection(
    ref: str,
    own_id: str,
    room_ids: set[str],
    type_to_first_id: dict[str, str],
) -> str | None:
    """Resolve a single connection reference to a concrete room id, or None."""
    if ref in room_ids:
        return ref
    if ref in type_to_first_id:
        return type_to_first_id[ref]
    return None


def graph_to_dict(G: nx.Graph) -> dict:
    """
    Serialize a graph into the {"nodes": [...], "edges": [...]} shape
    described in PHASE_1.md section 6, suitable for JSON API responses
    and for the frontend's vis-network rendering.
    """
    return {
        "nodes": [
            {"id": node_id, **attrs} for node_id, attrs in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v} for u, v in G.edges()
        ],
    }


def render_graph(G: nx.Graph, output_path: str, title: str = "Bubble Diagram") -> None:
    """
    Render the graph as a bubble diagram PNG using NetworkX + Matplotlib,
    per PHASE_1.md section 7. Node labels show "type (id)" so rooms of the
    same type remain distinguishable in the rendered image.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless rendering, no display needed
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)  # seeded for reproducible layout

    labels = {
        node_id: f"{attrs.get('type', node_id)}\n({node_id})"
        for node_id, attrs in G.nodes(data=True)
    }

    nx.draw(
        G,
        pos,
        ax=ax,
        with_labels=True,
        labels=labels,
        node_color="#A8D5E2",
        node_size=1800,
        font_size=8,
        edge_color="#555555",
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)