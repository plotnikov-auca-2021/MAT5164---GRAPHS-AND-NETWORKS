# Problem 5: Perfect matching on the induced subgraph over {A,B,C,D,E,F}
# with bipartition ({A,C,E}) vs ({B,D,F}), using Hopcroft-Karp.

import collections

edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]
U = ['A', 'C', 'E']
V = ['B', 'D', 'F']


def bipartite_cross(U, V, edges):
    E = collections.defaultdict(list)
    W = {}
    for u, v, w in edges:
        if u in U and v in V:
            E[u].append(v);
            W[(u, v)] = w
        elif v in U and u in V:
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

    m = 0
    while bfs():
        for u in U:
            if pairU[u] is None and dfs(u):
                m += 1
    return pairU, pairV, m


if __name__ == '__main__':
    E, W = bipartite_cross(U, V, edges)
    print("Cross edges with weights:")
    for u in U:
        for v in E.get(u, []):
            print(f"  {u} - {v} : {W[(u, v)]}")
    pairU, pairV, m = hopcroft_karp(E, U, V)
    match = [(u, v) for u, v in pairU.items() if v is not None]
    print("\nMatching size:", m)
    print("Matching:", match)
    print("Perfect?", m == 3)
