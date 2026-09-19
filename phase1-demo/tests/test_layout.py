from app.layout import build_layout
from app.graph import build_graph
from app.schema import ApartmentState


def test_empty_layout():
    state = ApartmentState(
        rooms=[],
        requirements={
            "bedrooms": 0,
            "bathrooms": 0,
            "kitchen": 0,
            "living_room": 0,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    assert placements == []


def test_single_room():
    state = ApartmentState(
        rooms=[
            {
                "id": "bedroom_1",
                "type": "bedroom",
                "count": 1,
                "size": "medium",
                "connections": [],
            }
        ],
        requirements={
            "bedrooms": 1,
            "bathrooms": 0,
            "kitchen": 0,
            "living_room": 0,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    assert len(placements) == 1

    room = placements[0]

    assert room.id == "bedroom_1"
    assert room.type == "bedroom"
    assert room.width == 4.0
    assert room.height == 4.0
    assert room.x == 0.0
    assert room.y == 0.0


def test_connected_rooms():
    state = ApartmentState(
        rooms=[
            {
                "id": "living_room_1",
                "type": "living_room",
                "count": 1,
                "size": "medium",
                "connections": ["kitchen_1"],
            },
            {
                "id": "kitchen_1",
                "type": "kitchen",
                "count": 1,
                "size": "medium",
                "connections": ["living_room_1"],
            },
        ],
        requirements={
            "bedrooms": 0,
            "bathrooms": 0,
            "kitchen": 1,
            "living_room": 1,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    assert len(placements) == 2

    living_room = next(
        room for room in placements
        if room.id == "living_room_1"
    )

    kitchen = next(
        room for room in placements
        if room.id == "kitchen_1"
    )

    assert living_room.x == 0.0
    assert living_room.y == 0.0

    # Kitchen should be placed to the right of the living room.
    assert kitchen.x == living_room.x + living_room.width
    assert kitchen.y == living_room.y


def test_room_sizes():
    state = ApartmentState(
        rooms=[
            {
                "id": "bedroom_1",
                "type": "bedroom",
                "count": 1,
                "size": "small",
                "connections": [],
            },
            {
                "id": "bedroom_2",
                "type": "bedroom",
                "count": 1,
                "size": "large",
                "connections": [],
            },
        ],
        requirements={
            "bedrooms": 2,
            "bathrooms": 0,
            "kitchen": 0,
            "living_room": 0,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    small = next(
        room for room in placements
        if room.id == "bedroom_1"
    )

    large = next(
        room for room in placements
        if room.id == "bedroom_2"
    )

    assert small.width == 3.0
    assert small.height == 3.0

    assert large.width == 5.0
    assert large.height == 4.0


def test_disconnected_rooms():
    state = ApartmentState(
        rooms=[
            {
                "id": "bedroom_1",
                "type": "bedroom",
                "count": 1,
                "size": "medium",
                "connections": [],
            },
            {
                "id": "kitchen_1",
                "type": "kitchen",
                "count": 1,
                "size": "medium",
                "connections": [],
            },
        ],
        requirements={
            "bedrooms": 1,
            "bathrooms": 0,
            "kitchen": 1,
            "living_room": 0,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    assert len(placements) == 2

    bedroom = next(
        room for room in placements
        if room.id == "bedroom_1"
    )

    kitchen = next(
        room for room in placements
        if room.id == "kitchen_1"
    )

    # They should not occupy the same position.
    assert (bedroom.x, bedroom.y) != (kitchen.x, kitchen.y)


def test_all_rooms_are_placed():
    state = ApartmentState(
        rooms=[
            {
                "id": "living_room_1",
                "type": "living_room",
                "count": 1,
                "size": "large",
                "connections": ["kitchen_1"],
            },
            {
                "id": "kitchen_1",
                "type": "kitchen",
                "count": 1,
                "size": "medium",
                "connections": ["living_room_1"],
            },
            {
                "id": "bedroom_1",
                "type": "bedroom",
                "count": 1,
                "size": "medium",
                "connections": [],
            },
            {
                "id": "bathroom_1",
                "type": "bathroom",
                "count": 1,
                "size": "small",
                "connections": [],
            },
        ],
        requirements={
            "bedrooms": 1,
            "bathrooms": 1,
            "kitchen": 1,
            "living_room": 1,
        },
    )

    graph = build_graph(state)
    placements = build_layout(state, graph)

    assert len(placements) == len(state.rooms)

    placed_ids = {room.id for room in placements}
    expected_ids = {room.id for room in state.rooms}

    assert placed_ids == expected_ids