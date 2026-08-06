#!/usr/bin/env python3
"""Amino-acid composition of each module class against the non-photosynthetic background.

Recomputed on cost_master_proteins.csv so that the residue-level panels describe exactly the
families the correlation panels describe.  The earlier tables came from the legacy per-copy
pipelines with different gene sets, which is why psbC and apcA were absent from them.

Test: Mann-Whitney on the per-protein frequency of each residue, module against background,
Benjamini-Hochberg across the twenty residues within a comparison.
Output: aa_enrichment.tsv  (one row per residue per comparison)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, false_discovery_control

ROOT = Path("/home/yangyicheng/27.evolution")
HERE = Path(__file__).resolve().parent
PEP = ROOT / "26.orthofinder/protein_sequences_folder"
AA_COST = ROOT / "38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv"
STD = "ACDEFGHIKLMNPQRSTVWY"

RC = ["Photosystem II", "Photosystem I", "Cytochrome b6f"]
PETC5 = RC + ["ATP synthase", "NADH dehydrogenase (NDH-1)"]
ANT = ["Light-harvesting antenna / phycobiliproteins", "Phycobilisome"]
LINKERS = ["cpcC", "cpcD", "cpcG"]


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
    return {k.split(".")[0]: v for k, v in seqs.items()}


def main():
    seqs = read_proteomes()
    prot = pd.read_csv(HERE / "cost_master_proteins.csv")
    cost = pd.read_csv(AA_COST, sep="\t").set_index("aa")["Cyano averaged cost"]

    frac = []
    for _, r in prot.iterrows():
        s = seqs.get(r.wp)
        if s is None:
            continue
        n = sum(1 for c in s if c in STD)
        if n == 0:
            continue
        row = {a: s.count(a) / n for a in STD}
        row.update(unit=r.unit, pathway=r.pathway, wp=r.wp)
        frac.append(row)
    F = pd.DataFrame(frac)
    print(f"amino-acid frequencies for {len(F):,} proteins")

    bg = F[~F.pathway.isin(PETC5 + ANT)]
    groups = {
        "Electron transport chain (PSII, PSI, cyt b6f)": F[F.pathway.isin(RC)],
        "Light-harvesting antenna": F[F.pathway.isin(ANT) & ~F.unit.isin(LINKERS)],
        "Phycobilisome linkers": F[F.unit.isin(LINKERS)],
    }
    rows = []
    for name, g in groups.items():
        recs = []
        for a in STD:
            u = mannwhitneyu(g[a], bg[a])
            mg, mb = g[a].mean(), bg[a].mean()
            recs.append(dict(comparison=name, n_proteins=len(g), AA=a,
                             mean_group=mg, mean_background=mb,
                             log2FC=np.log2(mg / mb) if mg > 0 and mb > 0 else np.nan,
                             aa_avg_cost=cost[a], P=u.pvalue))
        d = pd.DataFrame(recs)
        d["p_BH"] = false_discovery_control(d.P, method="bh")
        rows.append(d)
        sig = int((d.p_BH < 0.05).sum())
        up = ", ".join(d.nlargest(3, "log2FC").AA)
        dn = ", ".join(d.nsmallest(3, "log2FC").AA)
        print(f"  {name:<46s} n={len(g):>4d}  {sig}/20 BH-significant  "
              f"enriched {up} | depleted {dn}")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(HERE / "aa_enrichment.tsv", sep="\t", index=False, float_format="%.6g")
    print(f"\nwrote aa_enrichment.tsv ({len(out)} rows), background = {len(bg):,} proteins")


if __name__ == "__main__":
    main()
