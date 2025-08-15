# Problem 4: Bipartite subgraph on (V1,V2), Hall's theorem check,
# maximum matching (Hopcroft-Karp), and a minimum vertex cover (Kőnig).
import collections

V = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]
V1 = ['A', 'C', 'E', 'G', 'I']
V2 = ['B', 'D', 'F', 'H', 'J']


def bipartite_edges(V1, V2, edges):
    E = collections.defaultdict(list)
    W = {}
    for u, v, w in edges:
        if u in V1 and v in V2:
            E[u].append(v);
            W[(u, v)] = w
        elif v in V1 and u in V2:
            E[v].append(u);
            W[(v, u)] = w
    return E, W


def hopcroft_karp(E, U, V):
    INF = 10 ** 9
    pairU = {u: None for u in U}
    pairV = {v: None for v in V}
    dist = {u: INF for u in U}
    from collections import deque
    def bfs():
        Q = deque()
        for u in U:
            if pairU[u] is None:
                dist[u] = 0;
                Q.append(u)
            else:
                dist[u] = INF
        D = INF
        while Q:
            u = Q.popleft()
            if dist[u] < D:
                for v in E.get(u, []):
                    if pairV[v] is None:
                        D = dist[u] + 1
                    else:
                        if dist[pairV[v]] == INF:
                            dist[pairV[v]] = dist[u] + 1
                            Q.append(pairV[v])
        return D != INF

    def dfs(u):
        for v in E.get(u, []):
            if pairV[v] is None or (dist[pairV[v]] == dist[u] + 1 and dfs(pairV[v])):
                pairU[u] = v;
                pairV[v] = u;
                return True
        dist[u] = 10 ** 9;
        return False

    matching = 0
    while bfs():
        for u in U:
            if pairU[u] is None and dfs(u):
                matching += 1
    return pairU, pairV, matching


def hall_check(E, U, V):
    import itertools
    for r in range(1, len(U) + 1):
        for S in itertools.combinations(U, r):
            N = set()
            for u in S:
                N.update(E.get(u, []))
            if len(N) < len(S):
                return False, (set(S), N)
    return True, None


def min_vertex_cover_from_matching(E, U, V, pairU, pairV):
    from collections import deque
    U_unmatched = [u for u in U if pairU[u] is None]
    reachableU = set(U_unmatched)
    reachableV = set()
    Q = deque(U_unmatched)
    while Q:
        u = Q.popleft()
        for v in E.get(u, []):
            if pairU[u] != v and v not in reachableV:
                reachableV.add(v)
                u2 = pairV.get(v)
                if u2 is not None and u2 not in reachableU:
                    reachableU.add(u2)
                    Q.append(u2)
    coverU = [u for u in U if u not in reachableU]
    coverV = [v for v in V if v in reachableV]
    return coverU, coverV


if __name__ == '__main__':
    E, W = bipartite_edges(V1, V2, edges)
    print("Bipartite edges (left->right) with weights:")
    for u in V1:
        for v in E.get(u, []):
            print(f"  {u} - {v} : {W[(u, v)]}")
    ok, worst = hall_check(E, V1, V2)
    print("\nHall's condition satisfied?", ok)
    if not ok:
        S, N = worst
        print("Counterexample S:", S, "N(S):", N, "|N(S)| - |S| =", len(N) - len(S))
    pairU, pairV, size = hopcroft_karp(E, V1, V2)
    print("\nMaximum matching size:", size)
    matching = [(u, v) for u, v in pairU.items() if v is not None]
    print("Matching:", matching)
    cu, cv = min_vertex_cover_from_matching(E, V1, V2, pairU, pairV)
    print("\nOne minimum vertex cover (size={}):".format(len(cu) + len(cv)))
    print("  Left:", cu)
    print("  Right:", cv)
