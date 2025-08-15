# Problem 2: Dijkstra from A to H with a detailed iteration trace.
import heapq

V = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]


def undirected_adj(edges):
    g = {v: {} for v in V}
    for u, v, w in edges:
        g[u][v] = min(w, g[u].get(v, float('inf')))
        g[v][u] = min(w, g[v].get(u, float('inf')))
    return g


def dijkstra_trace(src, tgt):
    g = undirected_adj(edges)
    D = {v: float('inf') for v in V}
    P = {v: None for v in V}
    D[src] = 0.0
    pq = [(0.0, src)]
    seen = set()
    iter_no = 0
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u);
        iter_no += 1
        relaxes = []
        for v, w in g[u].items():
            nd = d + w
            if nd < D[v]:
                relaxes.append((v, D[v], nd))
                D[v] = nd;
                P[v] = u
                heapq.heappush(pq, (nd, v))
        pretty = []
        for v, old, new in relaxes:
            if old < 1e18:
                pretty.append(f"{v}: {old:.1f}->{new:.1f}")
            else:
                pretty.append(f"{v}: inf->{new:.1f}")
        print(f"Iter {iter_no} | Popped={u} | Dist[u]={d:.1f} | Relaxations:", pretty)
        if u == tgt:
            break
    path = [];
    cur = tgt
    if D[tgt] == float('inf'):
        path = None
    else:
        while cur is not None:
            path.append(cur);
            cur = P[cur]
        path.reverse()
    return D, P, path


if __name__ == '__main__':
    D, P, path = dijkstra_trace('A', 'H')
    print("\nShortest A->H distance:", D['H'])
    print("Shortest A->H path:", " -> ".join(path) if path else None)
