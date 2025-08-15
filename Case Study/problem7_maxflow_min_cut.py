# Problem 7: Maximum flow from A to H (capacities=weights), Edmonds–Karp, and a min cut.

from collections import deque, defaultdict

edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]
S = 'A';
T = 'H'


def make_directed_caps(edges):
    cap = defaultdict(lambda: defaultdict(float))
    for u, v, w in edges:
        cap[u][v] += w
        cap[v][u] += w
    return cap


def edmonds_karp(s, t):
    cap = make_directed_caps(edges)
    flow = defaultdict(lambda: defaultdict(float))
    total = 0.0;
    step = 0
    while True:
        # BFS for shortest augmenting path
        parent = {s: None};
        q = deque([s]);
        visited = set([s])
        P = {}
        found = False
        while q and not found:
            u = q.popleft()
            # forward arcs
            for v in cap[u]:
                if cap[u][v] - flow[u][v] > 1e-9 and v not in visited:
                    visited.add(v);
                    P[v] = u;
                    q.append(v)
                    if v == t: found = True; break
            if not found:
                # reverse arcs
                for v in cap:
                    if flow[v][u] > 1e-9 and v not in visited:
                        visited.add(v);
                        P[v] = u;
                        q.append(v)
                        if v == t: found = True; break
        if not found: break
        # reconstruct path and bottleneck
        path = [];
        cur = t;
        bott = float('inf')
        while cur != s:
            u = P[cur]
            if cur in cap[u]:
                res = cap[u][cur] - flow[u][cur];
                direction = +1
            else:
                res = flow[cur][u];
                direction = -1
            bott = min(bott, res)
            path.append((u, cur, direction))
            cur = u
        path.reverse();
        step += 1
        print(f"Step {step} path:", " -> ".join([u for u, _, _ in path] + [t]))
        print("  residuals:", ["{:.1f}".format((cap[u][v] - flow[u][v]) if d > 0 else flow[v][u]) for u, v, d in path])
        print("  bottleneck =", "{:.1f}".format(bott))
        # augment
        for u, v, d in path:
            if d > 0:
                flow[u][v] += bott
            else:
                flow[v][u] -= bott
        total += bott
    # min cut via reachability in residual
    visited = set([s]);
    q = deque([s])
    while q:
        u = q.popleft()
        for v in cap[u]:
            if cap[u][v] - flow[u][v] > 1e-9 and v not in visited:
                visited.add(v);
                q.append(v)
        for v in cap:
            if flow[v][u] > 1e-9 and v not in visited:
                visited.add(v);
                q.append(v)
    Sset = visited;
    Tset = set(cap.keys()) - Sset
    cut = 0.0
    for u in cap:
        for v in cap[u]:
            if u in Sset and v in Tset:
                cut += cap[u][v]
    print("\nMax flow =", "{:.1f}".format(total))
    print("Min cut (S|T): S =", sorted(Sset), ", T =", sorted(Tset))
    print("Cut capacity =", "{:.1f}".format(cut))


if __name__ == '__main__':
    edmonds_karp(S, T)
