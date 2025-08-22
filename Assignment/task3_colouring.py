"""
Task 3 — Graph Colouring (Vertex & Edge)

This script will:
  - Load the 15-vertex graph from task1_graph.json (default),
    or rebuild it using Task 1's builder (auto-import),
    or rebuild directly from CSV if --csv is provided.
  - 3.1: Compute a greedy vertex colouring (no adjacent vertices share a colour).
  - 3.2: Compute a proper edge colouring by colouring the line graph L(G)
         with a greedy strategy, then mapping colours back to edges.

Outputs:
  - task3_vertex_colouring.json
  - task3_edge_colouring.json
  - task3_colouring.txt   (human-readable summary)
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import networkx as nx
from networkx.algorithms.coloring import greedy_color


# --------------------------- I/O: graph persistence ---------------------------

def load_graph_json(path: Path) -> nx.Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.Graph()
    for n in data["nodes"]:
        G.add_node(
            n["name"],
            lat=float(n.get("lat", 0.0)),
            lon=float(n.get("lon", 0.0)),
            station_id=str(n.get("station_id", "")),
            country=str(n.get("country", "")),
            state=str(n.get("state", "")),
        )
    for e in data["edges"]:
        u, v = e["u"], e["v"]
        w = float(e["weight"])
        if w > 0.0:
            G.add_edge(u, v, weight=w)
    return G


def persist_graph_json(G: nx.Graph, path: Path) -> None:
    data = {
        "nodes": [
            {
                "name": n,
                "lat": float(G.nodes[n].get("lat", 0.0)),
                "lon": float(G.nodes[n].get("lon", 0.0)),
                "station_id": str(G.nodes[n].get("station_id", "")),
                "country": str(G.nodes[n].get("country", "")),
                "state": str(G.nodes[n].get("state", "")),
            }
            for n in sorted(G.nodes())
        ],
        "edges": [
            {"u": u, "v": v, "weight": float(G.edges[u, v]["weight"])}
            for u, v in G.edges()
            if float(G.edges[u, v]["weight"]) > 0.0
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --------------------------- Build / ensure graph -----------------------------

def try_build_via_task1():
    """
    Try importing the Task 1 builder (same folder):
      from task1_graph_construction import load_electric_usca, build_connected_15, DATA_CSV
    Build the 15-vertex graph and return it.
    """
    try:
        from task1_graph_construction import (
            load_electric_usca, build_connected_15, DATA_CSV
        )
    except Exception as e:
        raise RuntimeError(
            "Could not import Task 1 builder. "
            "Either run Task 1 first to produce task1_graph.json, "
            "or call this script with --csv PATH."
        ) from e

    stations = load_electric_usca(DATA_CSV)
    G, _ = build_connected_15(stations)
    return G


def build_from_csv(csv_path: Path) -> nx.Graph:
    """
    Use Task 1 builder directly but with a user-provided CSV.
    """
    try:
        from task1_graph_construction import (
            load_electric_usca, build_connected_15
        )
    except Exception as e:
        raise RuntimeError(
            "Could not import Task 1 builder to rebuild from CSV. "
            "Make sure task1_graph_construction.py is in the same folder."
        ) from e

    stations = load_electric_usca(str(csv_path))
    G, _ = build_connected_15(stations)
    return G


def ensure_graph(graph_path: Path, csv_path: Path | None) -> nx.Graph:
    """
    Load task1_graph.json if present. Otherwise rebuild via Task 1 module.
    If that import fails and --csv is provided, rebuild from CSV.
    """
    if graph_path.exists():
        return load_graph_json(graph_path)

    # Try auto-build via Task 1 module
    try:
        G = try_build_via_task1()
        persist_graph_json(G, graph_path)
        return G
    except Exception:
        if csv_path is None:
            raise FileNotFoundError(
                f"{graph_path} not found, and auto-build was unavailable. "
                f"Provide --csv PATH or run Task 1 to create the JSON."
            )
        G = build_from_csv(csv_path)
        persist_graph_json(G, graph_path)
        return G


# ------------------------------ Colouring (3.1) ------------------------------

def greedy_vertex_colouring(G: nx.Graph, strategy: str = "largest_first") -> Dict[str, int]:
    """
    Return dict: node -> colour_index using NetworkX greedy_color.
    Strategies include: 'largest_first', 'smallest_last', 'DSATUR', etc.
    """
    return greedy_color(G, strategy=strategy)


def invert_groups(colouring: Dict[str, int]) -> Dict[int, List[str]]:
    groups: Dict[int, List[str]] = {}
    for v, c in colouring.items():
        groups.setdefault(c, []).append(v)
    for c in groups:
        groups[c] = sorted(groups[c])
    return dict(sorted(groups.items()))


def validate_vertex_colouring(G: nx.Graph, colouring: Dict[str, int]) -> bool:
    for u, v in G.edges():
        if colouring[u] == colouring[v]:
            return False
    return True


# ------------------------------ Edge colouring (3.2) -------------------------

def greedy_edge_colouring_via_linegraph(G: nx.Graph, strategy: str = "largest_first") -> Dict[Tuple[str, str], int]:
    """
    Proper edge colouring by colouring the line graph L(G).
    Map colours back to original edges as sorted tuples (u,v).
    """
    # Build line graph: nodes of L are edges of G (as 2-tuples)
    LG = nx.line_graph(G)
    ecolor = greedy_color(LG, strategy=strategy)

    # Normalize edge keys as sorted tuples
    mapping: Dict[Tuple[str, str], int] = {}
    for edge_as_node, col in ecolor.items():
        u, v = edge_as_node
        if (v, u) in G.edges:
            key = tuple(sorted((u, v)))
        else:
            key = (u, v) if (u, v) in G.edges else (v, u)
        mapping[key] = col
    return mapping


def validate_edge_colouring(G: nx.Graph, ecolor: Dict[Tuple[str, str], int]) -> bool:
    """
    For every vertex, incident edges must have pairwise distinct colours.
    """
    for v in G.nodes():
        seen = set()
        for u in G.neighbors(v):
            key = tuple(sorted((u, v)))
            c = ecolor[key]
            if c in seen:
                return False
            seen.add(c)
    return True


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=Path("task1_graph.json"),
                    help="Path to task1_graph.json (default: ./task1_graph.json)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="If task1_graph.json is missing, rebuild from this CSV using Task 1 builder.")
    ap.add_argument("--vstrategy", default="largest_first",
                    help="Greedy strategy for vertex colouring (default: largest_first).")
    ap.add_argument("--estrategy", default="largest_first",
                    help="Greedy strategy for edge colouring via line graph (default: largest_first).")
    args = ap.parse_args()

    G = ensure_graph(args.graph, args.csv)

    # ---------------- 3.1 Vertex colouring ----------------
    vcol = greedy_vertex_colouring(G, strategy=args.vstrategy)
    v_groups = invert_groups(vcol)
    ok_v = validate_vertex_colouring(G, vcol)

    v_payload = {
        "strategy": args.vstrategy,
        "num_colours": len(set(vcol.values())),
        "valid": bool(ok_v),
        "groups": {f"C{idx+1}": verts for idx, verts in enumerate(v_groups.values())},
        "colouring": {v: int(c) for v, c in vcol.items()},
    }
    Path("task3_vertex_colouring.json").write_text(json.dumps(v_payload, indent=2), encoding="utf-8")

    # ---------------- 3.2 Edge colouring ------------------
    ecol = greedy_edge_colouring_via_linegraph(G, strategy=args.estrategy)
    ok_e = validate_edge_colouring(G, ecol)
    # group edges by colour
    e_groups: Dict[int, List[Tuple[str, str]]] = {}
    for e, c in ecol.items():
        e_groups.setdefault(c, []).append(tuple(e))
    for c in e_groups:
        e_groups[c] = sorted(list({tuple(sorted(e)) for e in e_groups[c]}))

    e_payload = {
        "strategy": args.estrategy,
        "num_colours": len(e_groups),
        "valid": bool(ok_e),
        "groups": {f"E{idx+1}": [(u, v) for (u, v) in edges]
                   for idx, (_, edges) in enumerate(sorted(e_groups.items()))},
        "colouring": {f"{u}-{v}": int(col) for (u, v), col in sorted(ecol.items())},
        "delta_max_degree": max(dict(G.degree()).values()),
    }
    Path("task3_edge_colouring.json").write_text(json.dumps(e_payload, indent=2), encoding="utf-8")

    # ---------------- Text summary ------------------------
    with open("task3_colouring.txt", "w", encoding="utf-8") as f:
        print("=== Task 3: Graph Colouring ===", file=f)
        print(f"Vertices: {len(G.nodes())}, Edges: {len(G.edges())}", file=f)
        print("\n-- Vertex colouring --", file=f)
        print(f"Strategy: {args.vstrategy}", file=f)
        print(f"Used {v_payload['num_colours']} colours; valid: {v_payload['valid']}", file=f)
        for cname, verts in v_payload["groups"].items():
            print(f"  {cname}: {', '.join(verts)}", file=f)

        print("\n-- Edge colouring (via line graph) --", file=f)
        print(f"Strategy: {args.estrategy}", file=f)
        print(f"Used {e_payload['num_colours']} colours; valid: {e_payload['valid']}", file=f)
        for cname, edges in e_payload["groups"].items():
            pairs = [f"({u},{v})" for (u, v) in edges]
            print(f"  {cname}: {', '.join(pairs)}", file=f)

    print("Task 3 complete.")
    print("- Saved task3_vertex_colouring.json")
    print("- Saved task3_edge_colouring.json")
    print("- Saved task3_colouring.txt")


if __name__ == "__main__":
    main()
