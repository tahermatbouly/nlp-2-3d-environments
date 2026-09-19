"""
Standalone test suite for app/graph.py -- no LLM/Ollama/network needed,
since JSON -> graph is meant to be pure deterministic Python.

Run with: python test_graph.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.schema import ApartmentState, Room, Requirements
from app.graph import build_graph, build_graph_report, graph_to_dict, render_graph


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def test_phase1_demo_case():
    """The exact scenario from PHASE_1.md section 8's demo flow."""
    print("\n--- Test 1: PHASE_1.md demo scenario ---")
    state = ApartmentState(
        rooms=[
            Room(id="living_room", type="living_room", size="large",
                 connections=["kitchen", "bedroom_1", "bedroom_2"]),
            Room(id="kitchen", type="kitchen", connections=["living_room"]),
            Room(id="bedroom_1", type="bedroom", connections=["living_room", "bathroom"]),
            Room(id="bedroom_2", type="bedroom", connections=["living_room"]),
            Room(id="bathroom", type="bathroom", connections=["bedroom_1"]),
        ],
        requirements=Requirements(bedrooms=2, bathrooms=1, kitchen=1, living_room=1),
    )

    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("5 rooms -> 5 nodes", G.number_of_nodes() == 5)
    all_pass &= check("no unresolved-reference warnings", len(warnings) == 0)
    all_pass &= check("living_room <-> kitchen edge exists", G.has_edge("living_room", "kitchen"))
    all_pass &= check("living_room <-> bedroom_1 edge exists", G.has_edge("living_room", "bedroom_1"))
    all_pass &= check("bedroom_1 <-> bathroom edge exists", G.has_edge("bedroom_1", "bathroom"))
    all_pass &= check("bedroom_2 is NOT connected to bathroom (wasn't declared)",
                       not G.has_edge("bedroom_2", "bathroom"))
    all_pass &= check("bidirectional declarations collapse correctly (4 unique edges)",
                       G.number_of_edges() == 4)
    return all_pass


def test_type_based_connection_ambiguity():
    """
    The core ambiguity: schema.py allows connections to reference either an
    exact room id OR a bare room type. PHASE_1.md's own example uses bare
    types ("connections": ["kitchen", "corridor"]). This must resolve
    correctly, picking the first room of that type.
    """
    print("\n--- Test 2: type-based connection resolution ---")
    state = ApartmentState(
        rooms=[
            Room(id="bedroom_1", type="bedroom", connections=["bathroom"]),
            Room(id="bedroom_2", type="bedroom", connections=[]),
            Room(id="bathroom_a", type="bathroom", connections=[]),
            Room(id="bathroom_b", type="bathroom", connections=[]),
        ],
        requirements=Requirements(bedrooms=2, bathrooms=2),
    )

    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("no warnings for a valid type-based reference", len(warnings) == 0)
    all_pass &= check(
        "'bathroom' (a type, not an id) resolves to the FIRST matching room (bathroom_a)",
        G.has_edge("bedroom_1", "bathroom_a"),
    )
    all_pass &= check(
        "does NOT incorrectly connect to bathroom_b",
        not G.has_edge("bedroom_1", "bathroom_b"),
    )
    return all_pass


def test_unresolvable_reference_is_reported_not_silent():
    """A connection to a room that doesn't exist by id or type must be
    reported as a warning, not silently dropped with no trace."""
    print("\n--- Test 3: unresolvable reference is surfaced, not swallowed ---")
    state = ApartmentState(
        rooms=[
            Room(id="bedroom_1", type="bedroom", connections=["garage"]),  # "garage" doesn't exist
        ],
        requirements=Requirements(bedrooms=1),
    )
    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("1 node still created despite bad reference", G.number_of_nodes() == 1)
    all_pass &= check("0 edges (nothing to connect to)", G.number_of_edges() == 0)
    all_pass &= check("exactly 1 warning generated", len(warnings) == 1)
    all_pass &= check("warning mentions the unresolved room 'garage'", "garage" in warnings[0])
    return all_pass


def test_self_connection_is_reported_not_silent():
    """A room listing itself as a connection must not create a self-loop,
    and must be reported so the source data issue is visible."""
    print("\n--- Test 4: self-connection handled safely ---")
    state = ApartmentState(
        rooms=[
            Room(id="kitchen", type="kitchen", connections=["kitchen"]),
        ],
        requirements=Requirements(kitchen=1),
    )
    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("no self-loop edge created", not G.has_edge("kitchen", "kitchen"))
    all_pass &= check("0 edges total", G.number_of_edges() == 0)
    all_pass &= check("1 warning generated for the self-reference", len(warnings) == 1)
    return all_pass


def test_isolated_room_with_no_connections():
    """A room with an empty connections list must still appear as an
    isolated node, not be dropped from the graph."""
    print("\n--- Test 5: isolated room (empty connections) still becomes a node ---")
    state = ApartmentState(
        rooms=[
            Room(id="storage", type="storage", connections=[]),
        ],
        requirements=Requirements(),
    )
    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("isolated room is present as a node", "storage" in G.nodes)
    all_pass &= check("no warnings", len(warnings) == 0)
    all_pass &= check("no edges", G.number_of_edges() == 0)
    return all_pass


def test_empty_state():
    """An empty ApartmentState (interview just started) must produce an
    empty graph, not crash."""
    print("\n--- Test 6: empty state produces an empty graph ---")
    state = ApartmentState()
    G, warnings = build_graph_report(state)

    all_pass = True
    all_pass &= check("0 nodes", G.number_of_nodes() == 0)
    all_pass &= check("0 edges", G.number_of_edges() == 0)
    all_pass &= check("0 warnings", len(warnings) == 0)
    return all_pass


def test_graph_to_dict_shape():
    """graph_to_dict must match the {"nodes": [...], "edges": [...]} shape
    documented in PHASE_1.md section 6, and each node's attributes must be
    reachable for the frontend renderer."""
    print("\n--- Test 7: graph_to_dict output shape ---")
    state = ApartmentState(
        rooms=[
            Room(id="kitchen", type="kitchen", size="small", connections=["living_room"]),
            Room(id="living_room", type="living_room", connections=["kitchen"]),
        ],
        requirements=Requirements(kitchen=1, living_room=1),
    )
    G = build_graph(state)
    d = graph_to_dict(G)

    all_pass = True
    all_pass &= check("top-level 'nodes' and 'edges' keys present",
                       "nodes" in d and "edges" in d)
    all_pass &= check("2 node entries", len(d["nodes"]) == 2)
    all_pass &= check("1 edge entry", len(d["edges"]) == 1)
    kitchen_node = next(n for n in d["nodes"] if n["id"] == "kitchen")
    all_pass &= check("node carries its type/size attributes",
                       kitchen_node["type"] == "kitchen" and kitchen_node["size"] == "small")
    edge = d["edges"][0]
    all_pass &= check("edge references the correct room ids",
                       {edge["source"], edge["target"]} == {"kitchen", "living_room"})
    return all_pass


def test_rendering_produces_a_file():
    """render_graph must actually produce a viewable PNG file."""
    print("\n--- Test 8: render_graph produces an image file ---")
    state = ApartmentState(
        rooms=[
            Room(id="living_room", type="living_room", connections=["kitchen", "bedroom_1"]),
            Room(id="kitchen", type="kitchen", connections=["living_room"]),
            Room(id="bedroom_1", type="bedroom", connections=["living_room"]),
        ],
        requirements=Requirements(bedrooms=1, kitchen=1, living_room=1),
    )
    G = build_graph(state)
    out_path = "/tmp/test_bubble_diagram.png"
    render_graph(G, out_path, title="Test Diagram")

    all_pass = True
    all_pass &= check("PNG file was created", os.path.exists(out_path))
    all_pass &= check("PNG file is non-empty", os.path.getsize(out_path) > 0)
    return all_pass


if __name__ == "__main__":
    results = [
        test_phase1_demo_case(),
        test_type_based_connection_ambiguity(),
        test_unresolvable_reference_is_reported_not_silent(),
        test_self_connection_is_reported_not_silent(),
        test_isolated_room_with_no_connections(),
        test_empty_state(),
        test_graph_to_dict_shape(),
        test_rendering_produces_a_file(),
    ]
    print(f"\n{'='*50}")
    if all(results):
        print(f"ALL {len(results)} TEST GROUPS PASSED")
    else:
        failed = len(results) - sum(results)
        print(f"{failed} / {len(results)} TEST GROUPS FAILED")
        sys.exit(1)