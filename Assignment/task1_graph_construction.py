"""
Task 1 — Graph construction & fundamentals (15-vertex EV subgraph, connected)

This version avoids random "fail to connect" by *growing* a connected set:
  1) Pick a seed station with many nearby neighbors (geo window).
  2) Repeatedly attach the nearest not-yet-selected station within 550 km
     to some selected vertex that still has degree < 5.
  3) After 15 vertices are chosen, add the shortest remaining edges (≤550 km)
     while respecting the degree cap, then save artifacts.

Artifacts:
  - task1_graph.json
  - task1_edges_inline.txt
  - task1_adjacency.csv
  - task1_incidence_small.csv
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx

# ============================== CONFIG =================================
DATA_CSV = r"C:\Users\nikit\PycharmProjects\MAT5164---GRAPHS-AND-NETWORKS\Assignment\Electric and Alternative Fuel Charging Stations.csv"
OUT_DIR = Path(".")
RANDOM_SEED = 5164      # only for deterministic seed choice tie-breaks
N_VERTS = 15
RANGE_KM = 550.0
DEGREE_CAP = 5
# =======================================================================


@dataclass
class Station:
    row_id: int
    station_id: str
    lat: float
    lon: float
    country: str
    state: str


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    p = math.pi / 180.0
    dlat = (lat2 - lat1) * p
    dlon = (lon2 - lon1) * p
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))


# ------------------------- column detection ----------------------------
def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def _find_col(df: pd.DataFrame, candidates: List[str], must: bool = True) -> Optional[str]:
    norm_map = {_norm(c): c for c in df.columns}
    # exact
    for c in candidates:
        n = _norm(c)
        if n in norm_map:
            return norm_map[n]
    # heuristics
    norms = {c: _norm(c) for c in df.columns}
    key = candidates[0].lower()
    if "latitude" in key or key == "lat":
        for c, n in norms.items():
            if "lat" in n:
                return c
    if "longitude" in key or key in {"lon", "long"}:
        for c, n in norms.items():
            if "lon" in n or "long" in n:
                return c
    if "country" in key:
        for c, n in norms.items():
            if "country" in n:
                return c
    if "state" in key or "prov" in key:
        for c, n in norms.items():
            if "state" in n or "province" in n or "prov" in n:
                return c
    if "fuel" in key:
        for c, n in norms.items():
            if "fuel" in n and ("type" in n or "code" in n):
                return c
    if "id" in key:
        for c, n in norms.items():
            if n.endswith("id") or "stationid" in n or "objectid" in n or n == "id":
                return c
    if must:
        raise KeyError(f"Could not find a column for any of: {candidates}")
    return None


def load_electric_usca(csv_path: str) -> List[Station]:
    df = pd.read_csv(csv_path, low_memory=False)

    lat_col = _find_col(df, ["latitude", "lat"])
    lon_col = _find_col(df, ["longitude", "lon", "long"])
    fuel_col = _find_col(df, ["fuel_type", "fuel type", "fuel type code", "primary fuel type"], must=False)
    country_col = _find_col(df, ["country", "country code"])
    state_col = _find_col(df, ["state_prov", "state/province", "state province", "state", "province", "prov"], must=False)
    id_col = _find_col(df, ["station_id", "station id", "id", "objectid", "siteid"], must=False)

    # Clean lat/lon
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df[df[lat_col].notna() & df[lon_col].notna()].copy()

    # Keep US/Canada
    df[country_col] = df[country_col].astype(str)
    countries_keep = {"US", "USA", "United States", "CA", "CAN", "Canada"}
    df = df[df[country_col].isin(countries_keep)].copy()

    # Electric filter if available
    if fuel_col is not None:
        mask_elec = df[fuel_col].astype(str).str.lower().str.contains("elec|electric", regex=True, na=False)
        df = df[mask_elec].copy()

    # Deduplicate identical coordinates
    df.drop_duplicates(subset=[lat_col, lon_col], inplace=True)

    stations: List[Station] = []
    for idx, r in df.reset_index(drop=True).iterrows():
        sid = str(r[id_col]) if id_col and pd.notna(r[id_col]) else f"row{idx}"
        country = str(r[country_col])
        state = str(r[state_col]) if state_col and pd.notna(r[state_col]) else ""
        stations.append(
            Station(
                row_id=int(idx),
                station_id=sid,
                lat=float(r[lat_col]),
                lon=float(r[lon_col]),
                country=country,
                state=state,
            )
        )
    return stations


# ------------------ greedy connected 15-node builder -------------------
def _geo_window(stations: List[Station], seed: Station) -> List[Station]:
    """
    Fast prefilter: keep stations within a lat/lon window roughly covering ~±650 km
    around the seed (so we have enough candidates for chaining ≤550 km hops).
    """
    dlat = 6.0  # ~ 6 deg ~ 667 km
    cl = max(0.15, math.cos(math.radians(seed.lat)))  # avoid division by ~0
    dlon = 6.0 / cl
    lat_min, lat_max = seed.lat - dlat, seed.lat + dlat
    lon_min, lon_max = seed.lon - dlon, seed.lon + dlon
    return [s for s in stations if (lat_min <= s.lat <= lat_max and lon_min <= s.lon <= lon_max)]


def _pick_seed(stations: List[Station]) -> Station:
    """
    Pick a seed with many neighbors in the geo window (heuristic: maximize pool size).
    """
    # try several seeds, pick the one with the largest geo window pool
    best = None
    for k in np.linspace(0, len(stations) - 1, num=25, dtype=int):
        seed = stations[int(k)]
        pool = _geo_window(stations, seed)
        if best is None or len(pool) > best[0]:
            best = (len(pool), seed)
    return best[1]


def build_connected_15(stations: List[Station]) -> Tuple[nx.Graph, Dict[str, Station]]:
    """
    Grow a connected 15-vertex subgraph by always attaching the nearest feasible station.
    Returns: (Graph with nodes named 'A'..'O', label->Station map)
    """
    if len(stations) < N_VERTS:
        raise RuntimeError("Not enough stations to build the subgraph.")

    # 1) choose a good seed and a candidate pool around it
    seed = _pick_seed(stations)
    pool = _geo_window(stations, seed)
    # ensure uniqueness & keep a bit more stations just in case
    if len(pool) < 60:
        # if too small, expand window a bit
        pool = stations

    # 2) select set S (connected by construction)
    # pick the station in pool that's "center-ish" by lon (start west->east chain nicely)
    start = min(pool, key=lambda s: abs(s.lon - seed.lon))
    selected: List[Station] = [start]
    degrees: Dict[int, int] = {start.row_id: 0}
    edges: List[Tuple[int, int, float]] = []

    # helper: can we attach 'cand' to any current vertex under constraints?
    def best_attachment(cand: Station) -> Optional[Tuple[int, float]]:
        best = None
        for u in selected:
            if degrees[u.row_id] >= DEGREE_CAP:
                continue
            d = haversine_km(cand.lat, cand.lon, u.lat, u.lon)
            if d <= RANGE_KM and d > 1e-9:
                if best is None or d < best[1]:
                    best = (u.row_id, d)
        return best

    # 3) grow until 15
    remaining = {s.row_id: s for s in pool if s.row_id != start.row_id}
    while len(selected) < N_VERTS:
        # among all remaining stations, pick the one with the *smallest* distance
        # to the current connected set (and feasible attachment w.r.t. degree & range)
        candidate_choice = None  # (cand_station, attach_u, dist)
        for cand in remaining.values():
            attach = best_attachment(cand)
            if attach is None:
                continue
            u, d = attach
            if (candidate_choice is None) or (d < candidate_choice[2]):
                candidate_choice = (cand, u, d)

        if candidate_choice is None:
            # Could not find any station within 550 km that can still attach.
            # Widen pool fallback: try again with whole dataset once.
            if len(remaining) != len(stations) - 1:
                remaining = {s.row_id: s for s in stations if s.row_id not in {x.row_id for x in selected}}
                continue
            raise RuntimeError("Could not grow to 15 vertices under 550 km / degree≤5 constraints.")

        cand, u, d = candidate_choice
        # add node and edge u--cand
        selected.append(cand)
        degrees.setdefault(cand.row_id, 0)
        degrees[u] = degrees.get(u, 0) + 1
        degrees[cand.row_id] += 1
        edges.append((u, cand.row_id, float(round(d, 1))))
        remaining.pop(cand.row_id, None)

    # 4) add extra short edges among the 15 (still obey caps)
    # compute all pairs, sort by distance, add if <= 550 and deg caps allow (and not duplicate)
    selected_ids = [s.row_id for s in selected]
    coord = {s.row_id: (s.lat, s.lon) for s in selected}
    have_edge = {(min(u, v), max(u, v)) for (u, v, _) in edges}
    extra = []
    for i in range(len(selected_ids)):
        for j in range(i + 1, len(selected_ids)):
            u, v = selected_ids[i], selected_ids[j]
            if (min(u, v), max(u, v)) in have_edge:
                continue
            d = haversine_km(coord[u][0], coord[u][1], coord[v][0], coord[v][1])
            if d <= RANGE_KM and d > 1e-9:
                extra.append((u, v, d))
    extra.sort(key=lambda x: x[2])
    for u, v, d in extra:
        if degrees[u] < DEGREE_CAP and degrees[v] < DEGREE_CAP:
            edges.append((u, v, float(round(d, 1))))
            degrees[u] += 1
            degrees[v] += 1

    # 5) build NetworkX graph and relabel by longitude A..O
    G = nx.Graph()
    for s in selected:
        G.add_node(s.row_id, lat=s.lat, lon=s.lon, station_id=s.station_id,
                   country=s.country, state=s.state)
    for u, v, w in edges:
        if w > 0:
            G.add_edge(u, v, weight=w)

    # Sort by longitude for labels A..O
    order = sorted(G.nodes(), key=lambda n: G.nodes[n]["lon"])
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    name_map = {old: labels[i] for i, old in enumerate(order)}
    G2 = nx.relabel_nodes(G, name_map, copy=True)
    label2station = {labels[i]: next(s for s in selected if s.row_id == order[i]) for i in range(len(order))}

    return G2, label2station


# ------------------------------ outputs --------------------------------
def write_edges_inline(G: nx.Graph, outfile: Path) -> None:
    parts = []
    for u, v, d in sorted(G.edges(data="weight"),
                          key=lambda e: (min(e[0], e[1]), max(e[0], e[1]))):
        parts.append(f"({u},{v})={d}")
    outfile.write_text(", ".join(parts) + ".\n", encoding="utf-8")


def write_adjacency(G: nx.Graph, outfile: Path) -> None:
    nodes = sorted(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=int)
    for u, v in G.edges():
        A[idx[u], idx[v]] = 1
        A[idx[v], idx[u]] = 1
    pd.DataFrame(A, index=nodes, columns=nodes).to_csv(outfile, encoding="utf-8")


def write_incidence_small(G: nx.Graph, outfile: Path) -> None:
    nodes = ["A", "B", "C", "D"]
    for n in nodes:
        if n not in G.nodes:
            nodes = sorted(G.nodes())[:4]
            break
    H = G.subgraph(nodes)
    edges = list(H.edges())
    mat = []
    for v in nodes:
        row = [1 if v in e else 0 for e in edges]
        mat.append(row)
    cols = [f"e{i+1}={edges[i][0]}{edges[i][1]}" for i in range(len(edges))]
    pd.DataFrame(mat, index=nodes, columns=cols).to_csv(outfile, encoding="utf-8")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stations = load_electric_usca(DATA_CSV)
    if len(stations) < 200:
        print(f"Warning: only {len(stations)} candidate stations after filtering; "
              f"the subgraph may be regionally tight.")

    G, label2station = build_connected_15(stations)

    # defensive cleanup: drop any non-positive edges (shouldn't happen)
    for u, v, w in list(G.edges(data="weight")):
        if w is None or w <= 0:
            G.remove_edge(u, v)

    # save artifacts
    graph_json = {
        "nodes": [
            {
                "name": n,
                "lat": float(G.nodes[n]["lat"]),
                "lon": float(G.nodes[n]["lon"]),
                "station_id": label2station[n].station_id,
                "country": G.nodes[n]["country"],
                "state": G.nodes[n]["state"],
            }
            for n in sorted(G.nodes())
        ],
        "edges": [
            {"u": u, "v": v, "weight": float(G.edges[u, v]["weight"])}
            for u, v in G.edges()
        ]
    }
    (OUT_DIR / "task1_graph.json").write_text(json.dumps(graph_json, indent=2), encoding="utf-8")
    write_edges_inline(G, OUT_DIR / "task1_edges_inline.txt")
    write_adjacency(G, OUT_DIR / "task1_adjacency.csv")
    write_incidence_small(G, OUT_DIR / "task1_incidence_small.csv")

    print("Task 1 complete.")
    print(f"- Saved: {OUT_DIR/'task1_graph.json'}")
    print(f"- Saved: {OUT_DIR/'task1_edges_inline.txt'}")
    print(f"- Saved: {OUT_DIR/'task1_adjacency.csv'}")
    print(f"- Saved: {OUT_DIR/'task1_incidence_small.csv'}")


if __name__ == "__main__":
    main()
