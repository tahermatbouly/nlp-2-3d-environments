"""
caption_resplan.py
------------------
Build (caption, structured spec) pairs for ResPlan floor plans.

ResPlan is NOT an image dataset: ResPlan.pkl is a list of dicts holding Shapely
geometries (bedroom, bathroom, kitchen, living, door, wall, ...) plus a NetworkX
graph. So the structured spec is derived directly from that geometry (exact, no
vision model, no hallucination) and an LLM is only used to phrase the caption.

Usage (from your repo, e.g. phase1-demo/):
    python scripts/caption_resplan.py --pkl /path/to/ResPlan.pkl --inspect
    python scripts/caption_resplan.py --pkl /path/to/ResPlan.pkl --limit 20
    python scripts/caption_resplan.py --pkl /path/to/ResPlan.pkl --lang ar --limit 20
    python scripts/caption_resplan.py --pkl /path/to/ResPlan.pkl --no-llm --limit 20

Requires: pip install shapely networkx groq pydantic python-dotenv
Env / .env: GROQ_API_KEY, optional TEXT_MODEL, SCHEMA_CLASS
"""

import argparse
import itertools
import json
import os
import pickle
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Repo root + .env
# ---------------------------------------------------------------------------
def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "app").is_dir():
            return p
    return start


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from pydantic import BaseModel, ValidationError

# ---------------------------------------------------------------------------
# CONFIG -- edit to match your project
# ---------------------------------------------------------------------------
OUTPUT_FILE = REPO_ROOT / "data" / "captions" / "resplan_captions.jsonl"
DEFAULT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")
SCHEMA_CLASS = os.getenv("SCHEMA_CLASS", "ApartmentSpec")

# ResPlan geometry keys that become rooms (edit / extend as needed)
ROOM_KEYS = ["living", "bedroom", "kitchen", "bathroom", "balcony"]
# Keys that become "requirements" strings when present
FEATURE_KEYS = ["garden", "parking", "pool"]
# ResPlan name -> your room_type vocabulary (lock this against your RPLAN labels)
ROOM_MAP = {
    "living": "living_room",
    "bedroom": "bedroom",
    "kitchen": "kitchen",
    "bathroom": "bathroom",
    "balcony": "balcony",
}
SIZE_SMALL, SIZE_LARGE = 0.10, 0.20   # share of total room area

# ---------------------------------------------------------------------------
# Schema: real one if importable, else a fallback (ADJUST build_spec BELOW
# if your real field names differ from room_type / room_a / room_b ...)
# ---------------------------------------------------------------------------
try:
    import app.schema as _schema
    Spec = getattr(_schema, SCHEMA_CLASS)
    print(f"[schema] using app.schema.{SCHEMA_CLASS}")
except Exception as e:
    print(f"[schema] could not import app.schema.{SCHEMA_CLASS} ({e}); using fallback schema")

    class Room(BaseModel):
        room_type: str
        count: int = 1
        size: Optional[str] = None

    class Connection(BaseModel):
        room_a: str
        room_b: str
        relation: str = "adjacent"

    class Spec(BaseModel):
        rooms: List[Room]
        connections: List[Connection] = []
        requirements: List[str] = []


# ---------------------------------------------------------------------------
# Geometry -> structured spec
# ---------------------------------------------------------------------------
def polys(geom):
    """Flatten a Shapely geometry (or None) into a list of non-empty polygons."""
    if geom is None or getattr(geom, "is_empty", True):
        return []
    parts = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
    out = []
    for p in parts:
        out.extend(polys(p) if hasattr(p, "geoms") else ([p] if not p.is_empty else []))
    return out


def size_label(share: float) -> str:
    return "small" if share < SIZE_SMALL else "large" if share >= SIZE_LARGE else "medium"


def build_spec(plan: dict) -> dict:
    """Derive rooms / connections / requirements from one ResPlan dict."""
    rooms = []                                   # (type, idx, polygon)
    for key in ROOM_KEYS:
        for i, p in enumerate(polys(plan.get(key))):
            rooms.append((key, i, p))
    if not rooms:
        return {"rooms": [], "connections": [], "requirements": []}

    total = sum(p.area for _, _, p in rooms) or 1.0
    eps = 0.02 * (statistics.median(p.area for _, _, p in rooms) ** 0.5)  # scale-free tolerance

    # rooms: one entry per room *type* with count + size of the largest instance
    room_entries = []
    for key in ROOM_KEYS:
        inst = [p for k, _, p in rooms if k == key]
        if inst:
            room_entries.append({
                "room_type": ROOM_MAP.get(key, key),
                "count": len(inst),
                "size": size_label(max(p.area for p in inst) / total),
            })

    # connections: door touching two rooms => "door"; rooms sharing a boundary
    # without a door => "adjacent" (open plan)
    name = lambda k, i: ROOM_MAP.get(k, k) + (f"_{i + 1}" if sum(1 for r in rooms if r[0] == k) > 1 else "")
    seen = {}
    doors = polys(plan.get("door"))
    for d in doors:
        touched = [(k, i) for k, i, p in rooms if p.buffer(eps).intersects(d)]
        for (k1, i1), (k2, i2) in itertools.combinations(touched, 2):
            seen[frozenset([name(k1, i1), name(k2, i2)])] = "door"
    for (k1, i1, p1), (k2, i2, p2) in itertools.combinations(rooms, 2):
        pair = frozenset([name(k1, i1), name(k2, i2)])
        if pair not in seen and p1.buffer(eps).intersects(p2):
            seen[pair] = "adjacent"
    connections = []
    for pair, rel in seen.items():
        a, b = sorted(pair)
        connections.append({"room_a": a, "room_b": b, "relation": rel})

    requirements = [f"has {k}" for k in FEATURE_KEYS if polys(plan.get(k))]
    return {"rooms": room_entries, "connections": connections, "requirements": requirements}


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------
LANG_RULES = {
    "en": "English",
    "ar": "natural Egyptian Arabic (colloquial); keep room names understandable to a normal Egyptian speaker",
}

CAPTION_PROMPT = """You write short requests that a person would type to describe an apartment.
Given this structured description of a real floor plan, write a natural request
in {lang}, 2-4 sentences, covering the rooms, their counts, relative sizes and
which rooms connect. Use ONLY the facts provided. Do not invent rooms, sizes or
connections. Output the caption text only.

{spec_json}"""


def template_caption(spec: dict) -> str:
    parts = [f"{r['count']} {r['room_type'].replace('_', ' ')}{'s' if r['count'] > 1 else ''} ({r['size']})"
             for r in spec["rooms"]]
    conns = [f"{c['room_a']} - {c['room_b']} ({c['relation']})" for c in spec["connections"]]
    text = "An apartment with " + ", ".join(parts) + "."
    if conns:
        text += " Connections: " + "; ".join(conns) + "."
    if spec["requirements"]:
        text += " " + ", ".join(spec["requirements"]).capitalize() + "."
    return text


def llm_caption(client, model: str, spec: dict, lang: str, retries: int = 3):
    prompt = CAPTION_PROMPT.format(lang=LANG_RULES[lang], spec_json=json.dumps(spec, ensure_ascii=False, indent=1))
    last = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.4, max_tokens=600,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content.strip(), None
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return None, last


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_done(out_path: Path) -> set:
    done = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("valid"):
                        done.add(str(rec["id"]))
                except Exception:
                    pass
    return done


def inspect(plans):
    p = plans[0]
    print(f"{len(plans)} plans. Keys of plan[0]:")
    for k, v in p.items():
        n = len(polys(v)) if hasattr(v, "geom_type") else None
        print(f"  {k:12s} {type(v).__name__:20s} {'polygons=' + str(n) if n is not None else repr(v)[:60]}")
    g = p.get("graph")
    if g is not None:
        print("\ngraph nodes (first 5):", list(g.nodes(data=True))[:5])
        print("graph edges (first 5):", list(g.edges(data=True))[:5])
    print("\nbuild_spec(plan[0]) ->")
    print(json.dumps(build_spec(p), indent=1))


def main():
    ap = argparse.ArgumentParser(description="Caption ResPlan plans from their vector geometry.")
    ap.add_argument("--pkl", required=True, help="path to ResPlan.pkl")
    ap.add_argument("--split", default=None, help="optional path to split.json (adds a 'split' field)")
    ap.add_argument("--out", default=str(OUTPUT_FILE))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--lang", choices=["en", "ar"], default="en")
    ap.add_argument("--no-llm", action="store_true", help="template caption, no API call")
    ap.add_argument("--inspect", action="store_true", help="print structure of the first plan and exit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.3)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        plans = pickle.load(f)          # trusted file from the official Kaggle release
    if args.inspect:
        return inspect(plans)

    split_of = {}
    if args.split:
        for name, ids in json.load(open(args.split)).items():
            for i in ids:
                split_of[str(i)] = name

    client = None
    if not args.no_llm:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit("GROQ_API_KEY is not set (or use --no-llm).")
        from groq import Groq
        client = Groq(api_key=api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out_path)
    todo = [p for p in plans if str(p.get("id")) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"output: {out_path}\n{len(done)} already done, {len(todo)} to process")

    with out_path.open("a", encoding="utf-8") as f:
        for n, plan in enumerate(todo, 1):
            pid = str(plan.get("id"))
            spec_dict = build_spec(plan)
            error = None
            try:
                spec_dict = Spec.model_validate(spec_dict).model_dump()
            except ValidationError as e:
                error = f"schema: {e}"
            if not spec_dict.get("rooms"):
                error = error or "no rooms found"

            caption = None
            if error is None:
                if args.no_llm:
                    caption = template_caption(spec_dict)
                else:
                    caption, error = llm_caption(client, args.model, spec_dict, args.lang)
                    time.sleep(args.sleep)

            rec = {
                "id": pid, "split": split_of.get(pid), "lang": args.lang,
                "model": None if args.no_llm else args.model,
                "area": plan.get("area"),
                "caption": caption, "spec": spec_dict if error is None else None,
                "valid": error is None, "error": error,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{n}/{len(todo)}] id={pid} valid={rec['valid']}")


if __name__ == "__main__":
    main()