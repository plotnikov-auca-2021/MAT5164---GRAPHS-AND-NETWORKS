# Problem 3: Minimum Spanning Tree via Kruskal + Union-Find.

V = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
edges = [('A', 'B', 3.0), ('A', 'C', 3.6), ('A', 'D', 3.7), ('B', 'D', 4.8), ('B', 'E', 4.4), ('C', 'D', 5.7),
         ('C', 'F', 6.2), ('D', 'E', 4.9), ('D', 'G', 4.6), ('D', 'F', 6.3), ('E', 'H', 5.2), ('E', 'G', 2.8),
         ('F', 'G', 6.1), ('F', 'I', 5.9), ('G', 'H', 5.5), ('G', 'J', 4.1), ('G', 'I', 4.2), ('H', 'J', 3.9),
         ('I', 'J', 3.4)]


class DSU:
    def __init__(self, items):
        self.p = {x: x for x in items}
        self.r = {x: 0 for x in items}

    def find(self, x):
        if self.p[x] != x:
            self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, a, b):
        pa, pb = self.find(a), self.find(b)
        if pa == pb: return False
        ra, rb = self.r[pa], self.r[pb]
        if ra < rb:
            self.p[pa] = pb
        elif rb < ra:
            self.p[pb] = pa
        else:
            self.p[pb] = pa; self.r[pa] += 1
        return True


def kruskal_mst(edges):
    dsu = DSU(V)
    mst = []
    for u, v, w in sorted(edges, key=lambda e: e[2]):
        if dsu.union(u, v):
            mst.append((u, v, w))
    total = sum(w for _, _, w in mst)
    return mst, total


if __name__ == '__main__':
    mst, total = kruskal_mst(edges)
    print("MST edges (Kruskal order):")
    for u, v, w in mst:
        print(f"  {u} - {v} : {w}")
    print("Total MST weight:", total)
