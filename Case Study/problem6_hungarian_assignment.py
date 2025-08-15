# Problem 6: 3x3 Assignment with Hungarian Algorithm. Agents X = {A,C,E}, Tasks Y = {B,D,F}.

import heapq

V = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]
X = ['A', 'C', 'E']
Y = ['B', 'D', 'F']


def undirected_adj(edges):
    g = {v: {} for v in V}
    for u, v, w in edges:
        g[u][v] = min(w, g[u].get(v, float('inf')))
        g[v][u] = min(w, g[v].get(u, float('inf')))
    return g


def dijkstra(src):
    g = undirected_adj(edges)
    D = {v: float('inf') for v in V}
    P = {v: None for v in V}
    D[src] = 0.0
    pq = [(0.0, src)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        for v, w in g[u].items():
            nd = d + w
            if nd < D[v]:
                D[v] = nd;
                P[v] = u
                heapq.heappush(pq, (nd, v))
    return D, P


def cost_matrix():
    mat = [[0.0] * 3 for _ in range(3)]
    for i, x in enumerate(X):
        D, _ = dijkstra(x)
        for j, y in enumerate(Y):
            mat[i][j] = D[y]
    return mat


def print_mat(tag, M):
    print(f"\n{tag}:")
    for i, row in enumerate(M):
        print("  ", ["{:.1f}".format(x) for x in row])


def hungarian_3x3(C):
    M = [row[:] for row in C]
    for i in range(3):
        rmin = min(M[i])
        for j in range(3):
            M[i][j] -= rmin
    print_mat("After row reduction", M)
    for j in range(3):
        cmin = min(M[i][j] for i in range(3))
        for i in range(3):
            M[i][j] -= cmin
    print_mat("After column reduction", M)

    def cover_zeros(M):
        zeros = [(i, j) for i in range(3) for j in range(3) if abs(M[i][j]) < 1e-9]
        best = (9, None)
        for mr in range(1 << 3):
            for mc in range(1 << 3):
                lines = bin(mr).count("1") + bin(mc).count("1")
                if lines >= best[0]: continue
                ok = True
                for i, j in zeros:
                    if not (((mr >> i) & 1) or ((mc >> j) & 1)):
                        ok = False;
                        break
                if ok:
                    best = (lines, (mr, mc))
        return best

    while True:
        lines, (mr, mc) = cover_zeros(M)
        print(f"Lines covering zeros: {lines} (rows={mr:03b}, cols={mc:03b})")
        if lines == 3: break
        m = min(M[i][j] for i in range(3) for j in range(3)
                if not ((mr >> i) & 1) and not ((mc >> j) & 1))
        for i in range(3):
            for j in range(3):
                covered = ((mr >> i) & 1) + ((mc >> j) & 1)
                if covered == 0:
                    M[i][j] -= m
                elif covered == 2:
                    M[i][j] += m
        print_mat(f"Adjust by m={m:.1f}", M)

    best = (float('inf'), None)
    used_cols = [False] * 3
    assign = [None] * 3

    def bt(k):
        nonlocal best
        if k == 3:
            cost = sum(C[i][assign[i]] for i in range(3))
            if cost < best[0]:
                best = (cost, assign[:])
            return
        for j in range(3):
            if abs(M[k][j]) < 1e-9 and not used_cols[j]:
                used_cols[j] = True;
                assign[k] = j
                bt(k + 1)
                used_cols[j] = False;
                assign[k] = None

    bt(0)
    return best


if __name__ == '__main__':
    C = cost_matrix()
    print_mat("Original cost matrix (shortest paths)", C)
    best_cost, assignment = hungarian_3x3(C)
    print("\nOptimal assignment (row->col):", assignment, "(rows A,C,E; cols B,D,F)")
    mapping = {0: 'B', 1: 'D', 2: 'F'}
    agents = ['A', 'C', 'E']
    for i, j in enumerate(assignment):
        print(f"  {agents[i]} -> {mapping[j]}  cost = {C[i][j]:.1f}")
    print("Total cost =", "{:.1f}".format(best_cost))
