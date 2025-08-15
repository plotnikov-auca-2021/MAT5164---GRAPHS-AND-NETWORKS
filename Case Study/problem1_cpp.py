# Problem 1: Chinese Postman Problem for the given undirected graph (start/end at 'A')

import math, heapq

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


def degrees(edges):
    deg = {v: 0 for v in V}
    for u, v, w in edges:
        deg[u] += 1;
        deg[v] += 1
    return deg


def dijkstra_allpairs(edges):
    g = undirected_adj(edges)
    dist = {}
    prev = {}
    for s in V:
        D = {v: float('inf') for v in V}
        P = {v: None for v in V}
        D[s] = 0.0
        pq = [(0.0, s)]
        seen = set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            for v, w in g[u].items():
                nd = d + w
                if nd < D[v]:
                    D[v] = nd
                    P[v] = u
                    heapq.heappush(pq, (nd, v))
        dist[s] = D
        prev[s] = P
    return dist, prev


def recover_path(prev_s, t):
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev_s[cur]
    return list(reversed(path))


def euler_tour_multigraph(multiedges, start):
    G = {u: dict(vs) for u, vs in multiedges.items()}
    for u in list(G):
        for v in list(G[u]):
            if G[u][v] == 0:
                del G[u][v]
    stack = [start]
    circuit = []
    while stack:
        u = stack[-1]
        if G[u]:
            v = next(iter(G[u]))
            G[u][v] -= 1
            if G[u][v] == 0: del G[u][v]
            G[v][u] -= 1
            if G[v][u] == 0: del G[v][u]
            stack.append(v)
        else:
            circuit.append(stack.pop())
    circuit.reverse()
    return circuit


def add_path_to_multigraph(M, path):
    for a, b in zip(path, path[1:]):
        M[a][b] = M[a].get(b, 0) + 1
        M[b][a] = M[b].get(a, 0) + 1


def cpp():
    deg = degrees(edges)
    odd = [v for v in V if deg[v] % 2 == 1]
    dist, prev = dijkstra_allpairs(edges)

    def best_matching(S):
        if not S:
            return 0.0, []
        a = S[0]
        best = (float('inf'), None)
        for i in range(1, len(S)):
            b = S[i]
            d = dist[a][b]
            rest = S[1:i] + S[i + 1:]
            cost, pairs = best_matching(rest)
            tot = d + cost
            if tot < best[0]:
                best = (tot, [(a, b)] + pairs)
        return best

    match_cost, pairs = best_matching(sorted(odd))
    M = {v: {} for v in V}
    for u, v, w in edges:
        M[u][v] = M[u].get(v, 0) + 1
        M[v][u] = M[v].get(u, 0) + 1
    for a, b in pairs:
        path = recover_path(prev[a], b)
        add_path_to_multigraph(M, path)
    tour = euler_tour_multigraph(M, 'A')
    base_sum = sum(w for _, _, w in edges)
    cpp_cost = base_sum + match_cost
    return {
        'odd_vertices': odd,
        'matching_pairs': pairs,
        'matching_cost': match_cost,
        'base_sum': base_sum,
        'cpp_cost': cpp_cost,
        'euler_tour_from_A': tour
    }


if __name__ == '__main__':
    out = cpp()
    print("Odd vertices:", out['odd_vertices'])
    print("Matching pairs and cost:", out['matching_pairs'], "cost =", out['matching_cost'])
    print("Base sum:", out['base_sum'])
    print("CPP cost:", out['cpp_cost'])
    print("Euler tour (A ... A):")
    print(" - ".join(out['euler_tour_from_A']))
