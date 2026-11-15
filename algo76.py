#!/usr/bin/env python3
"""
Algorithm 7.6 of S. Yang and X. Yu, "On lattice polarizable cubic fourfolds"
(arXiv:2103.09132v1), pp. 29-30, AI-generated implementation (Open Math OS, DeepSeek V4.1 Flash); not formally verified.

  input : two distinct integers d1, d2 with C_{d1}, C_{d2} non-empty Hassett divisors
  output: the rank-3 lattices whose M-polarizable loci are the irreducible
          components of C_{d1} ∩ C_{d2}

Three worked cases are run:
  (20,38) -> 17 components   [Thm 7.9,  pp. 30-31]   regression test
  (20,12) ->  6 components   [Thm 7.11, p.  32  ]   regression test (has an overlattice)
  ( 8,14) -> ?               [not treated in the paper]

The (8,14) answer printed at the end is a *computation*, not a quotation.

Usage:  python3 algo76.py [--check]
Dependencies: sympy, numpy.
"""
import sys
from fractions import Fraction as F
from itertools import combinations, product
from math import gcd

import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form

# ----------------------------------------------------------------- lattice data

def gram(case, n1, n2, tau):
    """Step (1): the Gram matrix of M_tau, in the basis (o_tau, mu_tau, nu_tau)."""
    if case == 'i':    # d1 = 6n1+2, d2 = 6n2+2, 1 <= n1 < n2
        return [[3, 1, 1], [1, 2 * n1 + 1, tau], [1, tau, 2 * n2 + 1]]
    if case == 'ii':   # d1 = 6n1+2, d2 = 6n2, 1 <= n1, 2 <= n2, tau >= 0
        return [[3, 1, 0], [1, 2 * n1 + 1, tau], [0, tau, 2 * n2]]
    if case == 'iii':  # d1 = 6n1, d2 = 6n2, 2 <= n1 < n2, tau >= 0
        return [[3, 0, 0], [0, 2 * n1, tau], [0, tau, 2 * n2]]
    raise ValueError(case)

def case_of(d):
    """Which of the three residue-class cases, and the corresponding n."""
    if d % 6 == 2:
        return 'i', (d - 2) // 6
    return 'ii_or_iii', d // 6

def setup(d1, d2):
    r1, r2 = d1 % 6, d2 % 6
    n1, n2 = (d1 - 2) // 6 if r1 == 2 else d1 // 6, (d2 - 2) // 6 if r2 == 2 else d2 // 6
    if r1 == 2 and r2 == 2:
        return 'i', n1, n2
    if r1 == 2 and r2 == 0:
        return 'ii', n1, n2
    if r1 == 0 and r2 == 0:
        return 'iii', n1, n2
    if r1 == 0 and r2 == 2:                     # swap so d1 = 6n1+2 when possible
        return 'ii', n2, n1
    raise ValueError((d1, d2))

# ------------------------------------------------------- rank-3 lattice toolkit

def dot(G, x, y):
    return sum(F(x[i]) * G[i][j] * F(y[j]) for i in range(3) for j in range(3))

class Lattice:
    """A rank-3 lattice by its integer Gram matrix; coordinates are in a fixed basis."""
    def __init__(self, G):
        self.G = [[int(v) for v in row] for row in G]
        M = Matrix(self.G)
        self.disc = abs(int(M.det()))
        self.dualcols = [tuple(F(M.inv()[i, j]) for i in range(3)) for j in range(3)]
        self._group()

    def bil(self, x, y):
        return dot(self.G, x, y)

    def _group(self):
        """M^v/M as a set of vectors mod 1, closed under addition."""
        zero = (F(0), F(0), F(0))
        E = {zero}
        for g in self.dualcols:
            E.add(tuple(c % 1 for c in g))
        changed = True
        while changed:
            changed = False
            for a in list(E):
                for g in self.dualcols:
                    s = tuple((a[i] + g[i]) % 1 for i in range(3))
                    if s not in E:
                        E.add(s); changed = True
        assert len(E) == self.disc, (len(E), self.disc)
        self.zero, self.els = zero, sorted(E)
        self.iso = [e for e in self.els if self.bil(e, e).denominator == 1]
        self.subgroups = self._isotropic_subgroups()

    def _isotropic_subgroups(self):
        """All H <= M^v/M with b(H,H) <= Z.  G(M) is a quotient of Z^3, so 3 generators suffice."""
        def closure(H, g):
            H = set(H)
            while True:
                new = {tuple((h[i] + g[i]) % 1 for i in range(3)) for h in H}
                if new <= H:
                    return frozenset(H)
                H |= new
        zero = self.zero
        good = {frozenset([zero])}
        frontier = [frozenset([zero])]
        for _ in range(3):                       # every subgroup of a 3-generated group
            nxt = []
            for H in frontier:
                for g in self.iso:
                    if g in H:
                        continue
                    H2 = closure(H, g)
                    if H2 in good:
                        continue
                    if all(self.bil(x, y).denominator == 1 for x in H2 for y in H2):
                        good.add(H2); nxt.append(H2)
            frontier = nxt
        return sorted(good, key=len)

    def overlattice(self, H):
        """Integral basis (and Gram matrix) of the overlattice N with N/M = H."""
        D = self.disc
        gens = [tuple(F(1) if i == j else F(0) for j in range(3)) for i in range(3)]
        gens += [g for g in sorted(H) if g != self.zero]
        cols = [[int(F(c) * D) for c in g] for g in gens]          # D*generators, integral
        Hn = hermite_normal_form(Matrix(cols).T)                   # columns span the same lattice
        B = []
        for j in range(Hn.cols):
            col = [Hn[i, j] for i in range(Hn.rows)]
            if any(c != 0 for c in col):
                B.append([F(c, D) for c in col])
        assert len(B) == 3
        Gn = [[self.bil(B[i], B[j]) for j in range(3)] for i in range(3)]
        assert all(e.denominator == 1 for row in Gn for e in row)
        return [[int(e) for e in row] for row in Gn], B

# ------------------------------------------------------------------- predicates

def vectors_of_small_norm(G, nmax):
    """All nonzero vectors of norm <= nmax, keyed by norm (positive definite G)."""
    lam = min(np.linalg.eigvalsh(np.array(G, dtype=float)))
    R = int(np.ceil(np.sqrt(nmax / lam))) + 1
    out = {}
    for x in product(range(-R, R + 1), repeat=3):
        if x == (0, 0, 0):
            continue
        n = sum(x[i] * G[i][j] * x[j] for i in range(3) for j in range(3))
        if 0 < n <= nmax:
            out.setdefault(n, []).append(x)
    return out

def has_root(G):
    return 2 in vectors_of_small_norm(G, 2)

def coords(B, v):
    Mx = Matrix([[F(B[j][i]) for j in range(3)] for i in range(3)])
    return [F(t) for t in Mx.solve(Matrix([F(t) for t in v]))]

def pair_is_primitive(B, v1, v2):
    """Is the rank-2 sublattice spanned by v1,v2 saturated in the lattice with basis B?"""
    c1, c2 = coords(B, v1), coords(B, v2)
    if any(c.denominator != 1 for c in c1 + c2):
        return False
    c1, c2 = [int(c) for c in c1], [int(c) for c in c2]
    g = 0
    for i, j in combinations(range(3), 2):
        g = gcd(g, abs(c1[i] * c2[j] - c1[j] * c2[i]))
    return g == 1

def o_perp_is_even(G, B):
    """Is <o>^perp even, where o is the first basis vector of the lattice with Gram G, basis B?"""
    c = coords(B, (F(1), F(0), F(0)))
    a = [sum(G[i][j] * c[j] for j in range(3)) for i in range(3)]
    assert all(x.denominator == 1 for x in a)
    a = [int(x) for x in a]
    g = 0
    for x in a:
        g = gcd(g, abs(x))
    if g == 0:
        return True
    a = [x // g for x in a]
    i0 = [i for i in range(3) if a[i] != 0][0]
    bas = []
    for j in range(3):
        if j == i0:
            continue
        z = [0, 0, 0]; z[j] = a[i0]; z[i0] = -a[j]
        bas.append(z)
    bas.append([bas[0][i] + bas[1][i] for i in range(3)])
    return all(sum(v[i] * G[i][j] * v[j] for i in range(3) for j in range(3)) % 2 == 0
               for v in bas)

# ------------------------------------------------------- Algorithm 7.6, verbatim

def run(d1, d2, verbose=True):
    case, n1, n2 = setup(d1, d2)
    if verbose:
        print(f"=== C_{d1} ∩ C_{d2}   case ({case}), n1={n1}, n2={n2}")

    # (2) T1 = {tau : disc(M_tau) > 0}   (positive definite by Sylvester's criterion)
    T1 = []
    for t in range(-400, 401):
        dsc = int(Matrix(gram(case, n1, n2, t)).det())
        if dsc > 0:
            T1.append((t, dsc))
    if case in ('ii', 'iii'):                 # Remark 7.7(a): M_tau = M_{-tau} here
        T1 = [x for x in T1 if x[0] >= 0]
    if verbose:
        print(f" (2) T1 = {[t for t, _ in T1]}")

    # (3) T2 = {tau in T1 : M_tau has no roots}
    T2 = []
    for t, dsc in T1:
        if not has_root(gram(case, n1, n2, t)):
            T2.append((t, dsc))
        elif verbose:
            print(f"     tau={t:3d} disc={dsc:4d}: M_tau has a root -> discarded")
    if verbose:
        print(f" (3) T2 = {[(t, d) for t, d in T2]}")

    # (4)+(5): overlattices N of M_tau, filtered, then up to isometry
    o, mu, nu = (F(1), F(0), F(0)), (F(0), F(1), F(0)), (F(0), F(0), F(1))
    out = []
    for t, dsc in T2:
        G = gram(case, n1, n2, t)
        L = Lattice(G)
        kept = [(1, G)]
        for H in L.subgroups:
            if len(H) == 1:
                continue
            Gn, B = L.overlattice(H)
            ok = (pair_is_primitive(B, o, mu) and pair_is_primitive(B, o, nu)
                  and o_perp_is_even(Gn, B) and not has_root(Gn))
            if ok:
                kept.append((len(H), Gn))
        if verbose:
            print(f" (4) tau={t:3d}  disc(M_tau)={dsc:4d}  "
                  f"{len(L.subgroups) - 1} nontrivial overlattice(s), admitted: "
                  f"{[abs(int(Matrix(g).det())) for h, g in kept if h > 1] or 'none'}")
        out += [(t, g) for _, g in kept]

    # (5) isometry classes: here distinct discriminants already separate them
    reps = {}
    for t, g in out:
        reps.setdefault(abs(int(Matrix(g).det())), (t, g))
    comps = [reps[d] for d in sorted(reps, reverse=True)]
    if verbose:
        print(f" (5) {len(comps)} component(s), discriminants {sorted(reps)}")
    return comps

CHECKS = {(20, 38): 17, (20, 12): 6}

if __name__ == '__main__':
    ok = True
    for (a, b), expected in CHECKS.items():
        comps = run(a, b)
        good = len(comps) == expected
        ok &= good
        print(f"    regression: |components| = {len(comps)}, paper says {expected}, "
              f"{'OK' if good else 'MISMATCH'}\n")
    comps = run(8, 14)
    print("\nComputed answer for C_8 ∩ C_14:")
    for tau, g in sorted(comps):
        print(f"    tau={tau:2d}   M_tau = [[{g[0][0]},{g[0][1]},{g[0][2]}],"
              f"[{g[1][0]},{g[1][1]},{g[1][2]}],[{g[2][0]},{g[2][1]},{g[2][2]}]]"
              f"   disc = {abs(int(Matrix(g).det()))}")
    print(f"    -> {len(comps)} irreducible components")
    if '--check' in sys.argv:
        print("\nregression checks:", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
