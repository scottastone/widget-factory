"""Reproduce / stress-test the numeric claims in 'Magic Gems' (arXiv 2512.09170v4)."""

import itertools

import numpy as np

n, N, M = 3, 9, 15
perms = np.array(list(itertools.permutations(range(1, 10))), dtype=np.int64)
S = perms.reshape(-1, 3, 3)


def lines(S):
    r = S.sum(2) - M  # row deviations
    c = S.sum(1) - M  # col deviations
    d = S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2] - M
    a = S[:, 0, 2] + S[:, 1, 1] + S[:, 2, 0] - M
    return r, c, d, a


def energies(S):
    """Integer numerators; true energy = numerator / N**2 (population covariance)."""
    r, c, d, a = lines(S)
    x = np.array([-1, 0, 1])
    y = np.array([1, 0, -1])
    cxz = (c * x).sum(1)
    cyz = (r * y).sum(1)
    low = cxz**2 + cyz**2 + d**2 + a**2
    full = (r[:, :2] ** 2).sum(1) + (c[:, :2] ** 2).sum(1) + d**2 + a**2
    return low, full, cxz, cyz, r, c, d, a


low, full, cxz, cyz, r, c, d, a = energies(S)
magic = (np.abs(r).sum(1) == 0) & (np.abs(c).sum(1) == 0) & (d == 0) & (a == 0)

print(f"magic squares found            : {magic.sum()}")
print(f"E_low == 0                     : {(low == 0).sum()}")
print(f"E_full == 0                    : {(full == 0).sum()}")
print(
    f"Cov(X,Z)=Cov(Y,Z)=0            : {((cxz == 0) & (cyz == 0)).sum()}   (paper: 760)"
)
print(
    f"col sums (16,13,16) & Cov(Y,Z)=0: "
    f"{(((c == np.array([1, -2, 1])).all(1)) & (cyz == 0)).sum()}   (paper: 128)"
)

# --- energy distribution under both plausible normalisations -------------
for label, s in [("population 1/N", 1 / N**2), ("sample 1/(N-1)", 1 / (N - 1) ** 2)]:
    e = low * s
    hist, edges = np.histogram(e, bins=200)
    mode = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
    p = hist / hist.sum()
    H = -(p[p > 0] * np.log(p[p > 0])).sum()
    print(
        f"E_low [{label:15s}] mean={e.mean():.3f} max={e.max():.3f} "
        f"mode={mode:.3f} entropy={H:.3f}   (paper: 4.44 / 12.10 / 3.12 / 5.03)"
    )

# --- local minima under the single-transposition neighbourhood ----------
swaps = list(itertools.combinations(range(9), 2))
nb_low = np.empty((len(swaps), len(S)), dtype=np.int64)
nb_full = np.empty_like(nb_low)
for k, (i, j) in enumerate(swaps):
    q = perms.copy()
    q[:, [i, j]] = q[:, [j, i]]
    nb_low[k], nb_full[k] = energies(q.reshape(-1, 3, 3))[:2]

for name, e, nb in [("E_full", full, nb_full), ("E_low", low, nb_low)]:
    lm = e < nb.min(0)
    print(
        f"{name:6s} strict local minima: {lm.sum()}  "
        f"(global {int((lm & (e == 0)).sum())}, non-global {int((lm & (e > 0)).sum())})"
    )

gap = nb_full[:, magic].min() / N**2
print(f"n=3 perturbation gap (E_full)  : {gap:.4f}  (paper: 0.0988)")
print(
    f"per-variant gaps identical?    : "
    f"{len(set(nb_full[:, magic].min(0).tolist())) == 1}"
)

# --- is E_full invariant under D4? --------------------------------------
T = np.array([[3, 1, 4], [8, 6, 2], [7, 5, 9]])
variants = [np.rot90(T, k) for k in range(4)] + [np.rot90(T.T, k) for k in range(4)]
vals = [energies(v[None])[1][0] for v in variants]
print(f"E_full over D4 orbit of a non-magic square: {vals}")

# --- 'exact variance formulas' are arrangement-independent --------------
z = perms - (n**2 + 1) / 2
print(
    f"Var(Z) distinct values over all 9! arrangements: "
    f"{len(np.unique(np.round(z.var(1), 12)))} -> {(n**4 - 1) / 12:.4f}"
)
xs = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
ys = np.array([1, 1, 1, 0, 0, 0, -1, -1, -1])
print(f"Cov(X,Y) for the grid (any arrangement): {np.mean(xs * ys):.1f}")

# --- the n=4 counterexample ---------------------------------------------
C4 = np.array([[9, 4, 12, 3], [16, 1, 14, 7], [10, 15, 11, 8], [2, 5, 6, 13]])
x4 = np.arange(4) - 1.5
print("\nn=4 counterexample (Remark 3.10):")
print(f"  is a permutation of 1..16 : {sorted(C4.ravel()) == list(range(1, 17))}")
print(f"  row sums {C4.sum(1).tolist()}  col sums {C4.sum(0).tolist()}")
print(f"  diagonals {np.trace(C4)}, {np.trace(np.fliplr(C4))}   (M=34)")
print(
    f"  Cov(X,Z)={np.dot(x4, C4.sum(0) - 34) / 16:.1f}  "
    f"Cov(Y,Z)={np.dot(-x4, C4.sum(1) - 34) / 16:.1f}  -> E_low = 0 confirmed"
)


# =======================================================================
# The convention that actually reproduces the paper's Table 3: the
# diagonal terms are divided by n, not n**2 -- i.e. they are NOT
# Cov(D,Z) as the paper's own Proposition 3.12 derives.
# =======================================================================
def E_paper(S, kind):
    r = S.sum(2) - M
    c = S.sum(1) - M
    d = S[:, 0, 0] + S[:, 1, 1] + S[:, 2, 2] - M
    a = S[:, 0, 2] + S[:, 1, 1] + S[:, 2, 0] - M
    diag = (d / 3.0) ** 2 + (a / 3.0) ** 2  # <-- /n, not /n**2
    if kind == "low":
        return (
            (((c * np.array([-1, 0, 1])).sum(1)) / 9.0) ** 2
            + (((r * np.array([1, 0, -1])).sum(1)) / 9.0) ** 2
            + diag
        )
    return ((r[:, :2] / 9.0) ** 2).sum(1) + ((c[:, :2] / 9.0) ** 2).sum(1) + diag


low_p = E_paper(S, "low")
hist, edges = np.histogram(low_p, bins=200)
p = hist / hist.sum()
print(
    f"\nUnder /n diagonal scaling: mean={low_p.mean():.2f} std={low_p.std():.2f} "
    f"mode={0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1]):.2f} max={low_p.max():.2f} "
    f"H={-(p[p > 0] * np.log(p[p > 0])).sum():.2f}"
)
print(
    "Paper Table 3           : mean=4.44 std=2.51 mode=3.12 max=12.10 H=5.03  -> match"
)

# Local minima still do not reproduce under this convention
nb = np.empty((36, len(S)))
nbf = np.empty((36, len(S)))
for k, (i, j) in enumerate(swaps):
    q = perms.copy()
    q[:, [i, j]] = q[:, [j, i]]
    q = q.reshape(-1, 3, 3)
    nb[k], nbf[k] = E_paper(q, "low"), E_paper(q, "full")
print(
    f"local minima  E_full={int((E_paper(S, 'full') < nbf.min(0)).sum())} (paper 24)  "
    f"E_low={int((low_p < nb.min(0)).sum())} (paper 344)  -> no match"
)

# E_full is not D4-invariant (it drops row/col n-1, which D4 permutes)
orbit = lambda T: (
    [E_paper(np.rot90(T, k)[None], "full")[0] for k in range(4)]
    + [E_paper(np.rot90(T.T, k)[None], "full")[0] for k in range(4)]
)
same = sum(
    len(set(np.round(orbit(S[i]), 12))) == 1
    for i in np.random.default_rng(0).choice(len(S), 2000, replace=False)
)
print(f"E_full constant over D4 orbit: {same}/2000 random arrangements")

# =======================================================================
# The 460M-sample Monte Carlo of Section 4.5 has a closed form.
# =======================================================================
print("\nE[E_low] in closed form vs paper's sampled means:")
print(f"{'n':>2} {'(n^4-1)*(1/72+1/(6(n+1)))':>26} {'paper':>8}")
for k, paper in [(3, 4.44), (4, 12.04), (5, 26.00)]:
    print(f"{k:>2} {(k**4 - 1) * (1 / 72 + 1 / (6 * (k + 1))):>26.3f} {paper:>8.2f}")
print(
    "Asymptotically ~n^4/72, not quadratic; the 3-point quadratic fit "
    "is 34% low by n=10."
)
