"""
Task 2 — Connectivity, Network Flows, and Shortest Path (A→O)

This script will:
  - Load the 15-vertex graph from task1_graph.json (default),
    or rebuild it using Task 1's builder (auto-import),
    or rebuild directly from CSV if --csv is provided.
  - 2.1: Report vertex connectivity (kappa), edge connectivity (lambda),
         and give one minimum node cut and one minimum edge cut.
  - 2.2: Build a directed unit-capacity version with supersource/supersink,
         then compute max-flow and a minimum s–t cut.
  - 2.3: Compute Dijkstra shortest path from A to O (weighted by km).

Outputs:
  - task2_connectivity.txt
  - task2_flow.json
  - task2_shortest_path.txt
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import networkx as nx


# --------------------------- helpers to load/build ----------------------------

def load_graph_json(path: Path) -> nx.Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.Graph()
    for n in data["nodes"]:
        G.add_node(
            n["name"],
            lat=float(n["lat"]),
            lon=float(n["lon"]),
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


def try_build_via_task1() -> Tuple[nx.Graph, Dict[str, dict]]:
    """
    Try importing the Task 1 builder (same folder):
      from task1_graph_construction import load_electric_usca, build_connected_15, DATA_CSV
    Build the 15-vertex graph and return it (already labeled A..O).
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
    G, label2station = build_connected_15(stations)
    return G, {k: vars(v) for k, v in label2station.items()}


def build_from_csv(csv_path: Path) -> nx.Graph:
    """
    Reuse Task 1 builder directly from its module, but pointing to a user-provided CSV.
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
    Load task1_graph.json if present. Otherwise rebuild from Task 1 builder
    (auto-import). If that import fails and --csv is provided, rebuild from CSV.
    """
    if graph_path.exists():
        return load_graph_json(graph_path)

    # Try auto-build via Task 1 module
    try:
        G, _ = try_build_via_task1()
        # also write the canonical JSON so other scripts see a consistent file
        persist_graph_json(G, graph_path)
        return G
    except Exception:
        if csv_path is None:
            raise FileNotFoundError(
                f"{graph_path} not found, and auto-build was unavailable. "
                f"Provide --csv PATH or run Task 1 to create the JSON."
            )
        # Rebuild from CSV using Task 1 functions
        G = build_from_csv(csv_path)
        persist_graph_json(G, graph_path)
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


# --------------------------- Task 2.1: connectivity ---------------------------

def connectivity_report(G: nx.Graph) -> Tuple[int, int, List[str], List[Tuple[str, str]]]:
    kappa = nx.node_connectivity(G)  # vertex connectivity
    lam = nx.edge_connectivity(G)  # edge connectivity

    # one minimum node cut and one minimum edge cut
    node_cut = list(nx.minimum_node_cut(G)) if kappa > 0 else []
    edge_cut = list(nx.minimum_edge_cut(G)) if lam > 0 else []
    return kappa, lam, sorted(node_cut), sorted(tuple(sorted(e)) for e in edge_cut)


# ----------------------- Task 2.2: max-flow / min-cut ------------------------

def build_directed_unit_capacity(Gu: nx.Graph,
                                 sources: Iterable[str],
                                 sinks: Iterable[str]) -> nx.DiGraph:
    D = nx.DiGraph()
    # undirected edge -> two arcs with capacity 1
    for u, v in Gu.edges():
        D.add_edge(u, v, capacity=1)
        D.add_edge(v, u, capacity=1)

    # supersource & supersink
    s, t = "_S", "_T"
    big = 10 ** 9
    for u in sources:
        D.add_edge(s, u, capacity=big)
    for v in sinks:
        D.add_edge(v, t, capacity=big)
    return D, s, t


def pick_sources_sinks(G: nx.Graph) -> Tuple[List[str], List[str]]:
    """
    Prefer {A,B} as sources and {N,O} as sinks if present.
    Otherwise choose two westernmost and two easternmost by longitude.
    """
    nodes = set(G.nodes())
    prefer_src = [n for n in ["A", "B"] if n in nodes]
    prefer_snk = [n for n in ["N", "O"] if n in nodes]
    if len(prefer_src) == 2 and len(prefer_snk) == 2:
        return prefer_src, prefer_snk

    # fallback: lon ranking
    have_lon = [n for n in nodes if "lon" in G.nodes[n]]
    if len(have_lon) >= 4:
        ranked = sorted(have_lon, key=lambda n: float(G.nodes[n].get("lon", 0.0)))
        return ranked[:2], ranked[-2:]
    # last resort: arbitrary
    ranked = sorted(nodes)
    return ranked[:2], ranked[-2:]


def run_max_flow(G: nx.Graph) -> Tuple[int, List[Tuple[str, str]]]:
    sources, sinks = pick_sources_sinks(G)
    D, s, t = build_directed_unit_capacity(G, sources, sinks)

    # Edmonds–Karp by default in networkx.maximum_flow
    flow_value, flow_dict = nx.maximum_flow(D, s, t)
    # Min cut (on the directed graph)
    cut_value, (S, T) = nx.minimum_cut(D, s, t)
    cut_edges = []
    for u in S:
        for v in D.successors(u):
            if v in T and D[u][v]["capacity"] > 0 and (u, v) != (s, t):
                if u in G.nodes() and v in G.nodes():
                    cut_edges.append(tuple(sorted((u, v))))
    cut_edges = sorted(set(cut_edges))
    return int(flow_value), cut_edges


# ---------------------- Task 2.3: Dijkstra A -> O ---------------------------

def dijkstra_A_to_O(G: nx.Graph) -> Tuple[List[str], float]:
    if "A" not in G or "O" not in G:
        raise ValueError("Graph does not contain labeled nodes 'A' and 'O'.")
    path = nx.shortest_path(G, "A", "O", weight="weight")
    dist = nx.path_weight(G, path, weight="weight")
    # round for nicer printing
    return path, round(float(dist), 3)


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=Path("task1_graph.json"),
                    help="Path to task1_graph.json (default: ./task1_graph.json)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="If task1_graph.json is missing, rebuild from this CSV using Task 1 builder.")
    args = ap.parse_args()

    G = ensure_graph(args.graph, args.csv)

    # --------- 2.1 connectivity ----------
    kappa, lam, node_cut, edge_cut = connectivity_report(G)
    with open("task2_connectivity.txt", "w", encoding="utf-8") as f:
        print(f"Vertex connectivity (kappa): {kappa}", file=f)
        print(f"Edge connectivity (lambda): {lam}", file=f)
        print(f"One minimum node cut: {node_cut}", file=f)
        print(f"One minimum edge cut: {edge_cut}", file=f)

    # --------- 2.2 max-flow/min-cut ------
    flow_value, mincut_edges = run_max_flow(G)
    flow_payload = {
        "max_flow": flow_value,
        "min_cut_edges": mincut_edges
    }
    Path("task2_flow.json").write_text(json.dumps(flow_payload, indent=2), encoding="utf-8")

    # --------- 2.3 Dijkstra A->O ---------
    try:
        sp_path, sp_len = dijkstra_A_to_O(G)
        with open("task2_shortest_path.txt", "w", encoding="utf-8") as f:
            print("Shortest path A->O (by km):", " - ".join(sp_path), file=f)
            print("Total length (km):", sp_len, file=f)
    except ValueError as e:
        with open("task2_shortest_path.txt", "w", encoding="utf-8") as f:
            print("Could not compute A->O:", e, file=f)

    print("Task 2 complete.")
    print("- Saved task2_connectivity.txt")
    print("- Saved task2_flow.json")
    print("- Saved task2_shortest_path.txt")


if __name__ == "__main__":
    main()
