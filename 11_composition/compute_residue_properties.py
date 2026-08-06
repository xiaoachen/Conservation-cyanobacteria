#!/usr/bin/env python3
"""Which residues are over-used by the more conserved families, and where in the structure.

One row per amino acid, one column per family property.  Each cell is the Spearman correlation,
across the 351 gene families, between that residue's mean relative content and that property.
Family level throughout: the earlier version of this table was per strain copy, which inflated n
about 36-fold and put P values as low as 1e-196 on coefficients of 0.3.

Two kinds of column, and they must not be read the same way:
  results  -- family CR, buried-core CR, surface CR, mean length
  scale    -- the four cost metrics.  Mean per-residue cost IS the composition-weighted average of
              the residue costs, so a residue's content correlates with it almost by construction.
              These columns say which residues define the cost axis; they are not findings.

Benjamini-Hochberg is applied across the whole table.

Inputs (read-only)
  cost_master.csv, cost_master_proteins.csv
  26.orthofinder/protein_sequences_folder/*.pep.faa
  40.whole_CR-MR/panorama_core_surface_351.csv
  38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv
Output
  residue_properties.tsv        20 residues x 11 properties, rho / P / P_BH / n
                                kind = result | scale | extra
  residue_content_by_family.csv one row per family, mean content of each residue
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, false_discovery_control

ROOT = Path("/home/yangyicheng/27.evolution")
HERE = Path(__file__).resolve().parent
PEP = ROOT / "26.orthofinder/protein_sequences_folder"
CS = ROOT / "40.whole_CR-MR/panorama_core_surface_351.csv"
AA_COST = ROOT / "38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv"
STD = "ACDEFGHIKLMNPQRSTVWY"

# (column in the joined table, label on the figure, kind)
PROPS = [("CR", "Family CR", "result"),
         ("core_CR", "Buried-core CR", "result"),
         ("surface_CR", "Surface CR", "result"),
         ("length", "Mean length", "result"),
         # scale columns.  Across the twenty residues, a residue's correlation with a cost metric
         # is largely predicted by its own value on that metric -- Spearman +0.88 for the
         # composite, +0.91 for weight, +0.80 to +0.84 for photon/carbon/ATP/NADPH -- and those
         # five metrics agree with each other at 0.96 to 0.99, so they are one column repeated.
         # Nitrogen demand is the exception (+0.38 against its own values, 0.09 to 0.26 against
         # the others) and is kept as the one demand column that is not a restatement.
         ("Cyano averaged cost", "CyanoAvg cost", "scale"),
         ("N assimilation demand", "N demand", "scale"),
         ("Growth-coupled photon cost", "Photon cost", "extra"),
         ("Inorganic carbon cost", "Inorganic C cost", "extra"),
         ("ATP turnover demand", "ATP demand", "extra"),
         ("NADPH turnover demand", "NADPH demand", "extra"),
         ("Weight", "Molecular weight", "extra")]


def read_proteomes():
    seqs = {}
    for f in sorted(PEP.glob("*.faa")):
        pid, buf = None, []
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                if pid:
                    seqs[pid] = "".join(buf)
                pid, buf = line[1:].split()[0], []
            else:
                buf.append(line.strip())
        if pid:
            seqs[pid] = "".join(buf)
    return {re.sub(r"\.\d+$", "", k): v for k, v in seqs.items()}


def main():
    seqs = read_proteomes()
    t = pd.read_csv(HERE / "cost_master.csv")
    prot = pd.read_csv(HERE / "cost_master_proteins.csv")

    rows = []
    for wp, unit in zip(prot.wp, prot.unit):
        s = seqs.get(wp)
        if s is None:
            continue
        n = sum(1 for c in s if c in STD)
        if n == 0:
            continue
        d = {a: s.count(a) / n for a in STD}
        d.update(unit=unit, wp=wp)
        rows.append(d)
    F = pd.DataFrame(rows)
    print(f"residue content for {len(F):,} proteins in {F.unit.nunique()} families")
    fam = F.groupby("unit")[list(STD)].mean()
    fam.reset_index().to_csv(HERE / "residue_content_by_family.csv", index=False,
                             float_format="%.6g")

    cs = pd.read_csv(CS).set_index("unit")[["core_CR", "surface_CR"]]
    length = prot.groupby("unit").length.mean().rename("length")
    # PROPS already lists CR, so build the column list once and de-duplicate: a repeated name
    # would make X[col] a DataFrame and every correlation silently NaN
    tcols = list(dict.fromkeys(p for p, _l, _k in PROPS if p in t.columns))
    X = fam.join(t.set_index("unit")[tcols]).join(cs).join(length)

    aa_cost = pd.read_csv(AA_COST, sep="\t").set_index("aa")["Cyano averaged cost"]
    recs = []
    for a in STD:
        for col, lab, kind in PROPS:
            q = X.dropna(subset=[col, a])
            r = spearmanr(q[a], q[col])
            recs.append(dict(AA=a, aa_cost=aa_cost[a], property=lab, kind=kind, n=len(q),
                             rho=r.statistic, P=r.pvalue))
    d = pd.DataFrame(recs)
    d["P_BH"] = false_discovery_control(d.P.values, method="bh")
    d.to_csv(HERE / "residue_properties.tsv", sep="\t", index=False, float_format="%.6g")

    # ---------------------------------------------------------------- core minus surface
    # The two correlations share the same families, so they are dependent and their P values
    # cannot be compared.  Bootstrapping families gives an interval on the difference itself,
    # which is what turns "the two columns look different" into a test.
    rng = np.random.default_rng(0)
    B = 4000
    Y = X.dropna(subset=["core_CR", "surface_CR"])
    idx = [rng.integers(0, len(Y), len(Y)) for _ in range(B)]
    diff = []
    for a in STD:
        av, cv, sv = Y[a].values, Y.core_CR.values, Y.surface_CR.values
        rc = spearmanr(av, cv).statistic
        rs = spearmanr(av, sv).statistic
        bs = np.array([spearmanr(av[i], cv[i]).statistic - spearmanr(av[i], sv[i]).statistic
                       for i in idx])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        pv = max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / B)
        diff.append(dict(AA=a, aa_cost=aa_cost[a], n=len(Y), core=rc, surface=rs,
                         diff=rc - rs, lo=lo, hi=hi, P=pv))
    dd = pd.DataFrame(diff)
    dd["P_BH"] = false_discovery_control(dd.P.values, method="bh")
    dd.to_csv(HERE / "residue_core_vs_surface.tsv", sep="\t", index=False, float_format="%.6g")
    sig = dd[dd.P_BH < .05]
    sig = sig.reindex(sig["diff"].abs().sort_values(ascending=False).index)
    print(f"\ncore minus surface, {B} bootstrap resamples of {len(Y)} families: "
          f"{len(sig)} of 20 residues differ (BH < 0.05)")
    for _, r in sig.iterrows():
        print(f"   {r.AA}  core {r.core:+.3f}  surface {r.surface:+.3f}  "
              f"delta {r['diff']:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}]  BH {r.P_BH:.4f}")

    res = d[d.kind == "result"]
    print(f"20 residues x {len(PROPS)} properties; "
          f"{int((d.P_BH < .05).sum())}/{len(d)} cells BH-significant "
          f"({int((res.P_BH < .05).sum())}/{len(res)} among the four result columns)")
    print("\nstrongest associations with family conservation:")
    cr = d[(d.property == "Family CR")].reindex(
        d[(d.property == "Family CR")].rho.abs().sort_values(ascending=False).index)
    for _, r in cr.head(6).iterrows():
        st = "***" if r.P_BH < 1e-3 else "**" if r.P_BH < 1e-2 else "*" if r.P_BH < .05 else "n.s."
        print(f"   {r.AA}  rho = {r.rho:+.3f}  {st:<4s}  (cost {r.aa_cost:.2f})")
    print("\ncore versus surface, largest difference:")
    piv = d[d.property.isin(["Buried-core CR", "Surface CR"])].pivot_table(
        index="AA", columns="property", values="rho")
    piv["diff"] = piv["Buried-core CR"] - piv["Surface CR"]
    for a, r in piv.reindex(piv["diff"].abs().sort_values(ascending=False).index).head(5).iterrows():
        print(f"   {a}  core {r['Buried-core CR']:+.3f}   surface {r['Surface CR']:+.3f}   "
              f"Δ {r['diff']:+.3f}")
    print("\nwrote residue_properties.tsv and residue_content_by_family.csv")


if __name__ == "__main__":
    main()
