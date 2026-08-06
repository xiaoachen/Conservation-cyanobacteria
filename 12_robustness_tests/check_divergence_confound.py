#!/usr/bin/env python3
"""Does divergence regression explain the antenna's negative cost-CR slope?

The concern, raised when the earlier copy-level antenna claim was withdrawn on 2026-08-02: CR is
measured against a reference, so a family that has diverged more has a lower CR.  If substitution
also drags composition back toward the proteome mean, then a *cheap* module drifts toward being
expensive as it diverges, and "low CR, high cost" appears with no economics behind it.  The antenna
is exactly a cheap module, so the confound has a target there and not in the transport core.

That objection was established at protein-copy level.  The finalised analysis is at family level,
so it has to be re-tested there before the antenna number can be written into the Results.

Operationalisation at family level: the family's mean composition is a point in the 20-simplex; the
background mean composition is another.  Euclidean distance between them is how far the family sits
from the proteome norm.  The confound predicts, inside the antenna, that lower CR goes with a
SMALLER distance (the composition has been pulled in) and that a smaller distance goes with HIGHER
cost (because the background is costlier than the antenna).  Both must hold for it to mediate, and
the partial correlation of cost against CR given distance must then collapse.

Limitation, stated rather than hidden: the copy-level version compared each target against its own
reference, so it measured movement.  At family level only the standing distance is available, since
cost_master_proteins.csv does not mark which member was the reference.  A standing distance is a
weaker instrument than a displacement, so a null here is weaker evidence than a null there would be.

Input  : cost_master.csv, residue_content_by_family.csv
Output : divergence_confound.tsv  (printed in full as well)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
STD = "ACDEFGHIKLMNPQRSTVWY"
COST = "Cyano averaged cost"
RC = ["Photosystem II", "Photosystem I", "Cytochrome b6f"]
PETC5 = RC + ["ATP synthase", "NADH dehydrogenase (NDH-1)"]
ANT = ["Light-harvesting antenna / phycobiliproteins", "Phycobilisome"]
LINKERS = ["cpcC", "cpcD", "cpcG"]


def partial_spearman(x, y, z):
    """Spearman correlation of x and y with z partialled out, on ranks."""
    R = np.column_stack([pd.Series(v).rank().values for v in (x, y, z)])
    zc = np.column_stack([np.ones(len(R)), R[:, 2]])
    b = np.linalg.lstsq(zc, R[:, :2], rcond=None)[0]
    res = R[:, :2] - zc @ b
    r = spearmanr(res[:, 0], res[:, 1])
    return r.statistic, r.pvalue


def main():
    t = pd.read_csv(HERE / "cost_master.csv")
    F = pd.read_csv(HERE / "residue_content_by_family.csv").set_index("unit")
    t["block"] = np.where(t.pathway.isin(RC), "PETC core",
                          np.where(t.pathway.isin(ANT) & ~t.unit.isin(LINKERS), "antenna",
                                   np.where(t.unit.isin(LINKERS), "linker",
                                            np.where(t.pathway.isin(PETC5), "other PETC",
                                                     "background"))))
    X = t.set_index("unit").join(F, how="inner")
    bg = X[X.block == "background"][list(STD)].mean()
    X["dist_bg"] = np.sqrt(((X[list(STD)] - bg) ** 2).sum(axis=1))

    rows = []
    for name, q in [("whole panel", X), ("background", X[X.block == "background"]),
                    ("PETC core (PSII+PSI+b6f)", X[X.block == "PETC core"]),
                    ("antenna, chromophore-binding", X[X.block == "antenna"]),
                    ("antenna + linkers", X[X.block.isin(["antenna", "linker"])])]:
        n = len(q)
        r_cost = spearmanr(q[COST], q.CR)
        r_dist = spearmanr(q.dist_bg, q.CR)
        r_dc = spearmanr(q.dist_bg, q[COST])
        pr, pp = partial_spearman(q[COST], q.CR, q.dist_bg) if n >= 5 else (np.nan, np.nan)
        rows.append(dict(block=name, n=n, rho_cost_CR=r_cost.statistic, P_cost_CR=r_cost.pvalue,
                         rho_dist_CR=r_dist.statistic, P_dist_CR=r_dist.pvalue,
                         rho_dist_cost=r_dc.statistic, P_dist_cost=r_dc.pvalue,
                         partial_cost_CR=pr, P_partial=pp))
    d = pd.DataFrame(rows)
    d.to_csv(HERE / "divergence_confound.tsv", sep="\t", index=False, float_format="%.4g")

    print(f"background mean composition from {int((X.block == 'background').sum())} families; "
          f"distance is Euclidean in the 20-residue simplex\n")
    hdr = (f"{'block':30s} {'n':>4s} {'cost x CR':>11s} {'dist x CR':>11s} "
           f"{'dist x cost':>12s} {'cost x CR | dist':>17s}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in d.iterrows():
        print(f"{r.block:30s} {int(r.n):4d} "
              f"{r.rho_cost_CR:+7.3f} {'*' if r.P_cost_CR < .05 else ' ':1s}  "
              f"{r.rho_dist_CR:+7.3f} {'*' if r.P_dist_CR < .05 else ' ':1s}  "
              f"{r.rho_dist_cost:+8.3f} {'*' if r.P_dist_cost < .05 else ' ':1s}  "
              f"{r.partial_cost_CR:+11.3f} {'*' if r.P_partial < .05 else ' ':1s}")
    print("\nthe confound requires dist x CR > 0 and dist x cost < 0 in the antenna, "
          "and the partial to collapse")
    a = d[d.block == "antenna, chromophore-binding"].iloc[0]
    print(f"antenna: dist x CR = {a.rho_dist_CR:+.3f} (P = {a.P_dist_CR:.3f}), "
          f"dist x cost = {a.rho_dist_cost:+.3f} (P = {a.P_dist_cost:.3f}), "
          f"cost x CR falls {a.rho_cost_CR:+.3f} -> {a.partial_cost_CR:+.3f} "
          f"(P = {a.P_partial:.3f})")


if __name__ == "__main__":
    main()
