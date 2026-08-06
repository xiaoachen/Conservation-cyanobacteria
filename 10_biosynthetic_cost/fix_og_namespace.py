#!/usr/bin/env python3
"""The 351-family cost table was built by looking up run-A orthogroup ids in run-B's membership.

`panorama_families.csv` carries two id namespaces at once: the 222 single-copy core families keep
the ids of the earlier OrthoFinder run (run A, the one the CR pipeline used), while the
photosynthetic families were re-keyed to Results_May21 (run B).  `compute_cost_351.py` reads
membership only from run B, so for every core-222 family it costed a different gene family --
run A's OG0000954 is run B's OG0000984, and so on.  CR was unaffected, which is why the error
was invisible: the rows looked right and only the cost column was wrong.

This script builds the run-A -> run-B translation from sequence membership, recomputes the cost
table, and prints the headline correlations before and after.  Read-only except for its two
outputs.
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/yangyicheng/27.evolution")
HERE = Path(__file__).resolve().parent
AA_COST = ROOT / "38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv"
PEP = ROOT / "26.orthofinder/protein_sequences_folder"
OGT = PEP / "OrthoFinder/Results_May21/Orthogroups/Orthogroups.tsv"
PAN = ROOT / "40.whole_CR-MR/panorama_families.csv"
PAIR = ROOT / "38.protein_cost/protein_cost_pairwise.csv"
CORE222 = ROOT / "26.orthofinder/single_copy_core_OGs_222.tsv"

METRICS = ["Cyano averaged cost", "Growth-coupled photon cost", "Inorganic carbon cost",
           "ATP turnover demand", "NADPH turnover demand", "N assimilation demand", "Weight"]
STD = set("ACDEFGHIKLMNPQRSTVWY")


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
    return seqs


def per_protein_cost(seq, aa):
    counts = defaultdict(int)
    n = 0
    for c in seq:
        if c in STD:
            counts[c] += 1
            n += 1
    if n == 0:
        return None
    tot = np.zeros(len(METRICS))
    for c, k in counts.items():
        tot += aa.loc[c].values * k
    return tot / n


def main():
    aa = pd.read_csv(AA_COST, sep="\t")
    aa = aa[aa.aa.isin(STD)].set_index("aa")[METRICS]
    seqs = read_proteomes()

    og = pd.read_csv(OGT, sep="\t", dtype=str).set_index("Orthogroup")
    members, of_og = {}, {}
    for g, r in og.iterrows():
        ids = []
        for cell in r.dropna():
            if str(cell).strip():
                ids += [p.strip() for p in str(cell).split(", ")]
        members[g] = ids
        for p in ids:
            of_og[re.sub(r"\.\d+$", "", p)] = g

    # ------------------------------------------------ the core-222 list is its own namespace
    # build_whole_panorama.py takes the orthogroup id straight from the CR filename whenever the
    # unit is named `OG*` (the 222 single-copy core families) and only maps WP -> orthogroup for
    # everything else.  Those filenames come from the earlier OrthoFinder run saved as
    # single_copy_core_OGs_222.tsv, so their membership must be read from that file, not from
    # Results_May21 -- the two runs number their orthogroups differently.
    core = pd.read_csv(CORE222, sep="\t", dtype=str).set_index("Orthogroup")
    core_members = {}
    for g, r in core.iterrows():
        ids = []
        for cell in r.dropna():
            if str(cell).strip():
                ids += [p.strip() for p in str(cell).split(", ")]
        core_members[g] = ids
    print(f"core-222 list: {len(core_members)} families from {CORE222.name}")

    pair = pd.read_csv(PAIR)
    sc = pair[pair.pathway == "01.singlecopy_gene"].copy()
    agree = sum(1 for g, ids in core_members.items()
                if set(sc[sc.gene == g].target_wp.astype(str))
                <= {re.sub(r"\.\d+$", "", p) for p in ids})
    print(f"  membership matches the pairwise CR table for {agree} of {len(core_members)}")
    same = sum(1 for g, ids in core_members.items()
               if {re.sub(r'\.\d+$', '', p) for p in ids}
               == {re.sub(r'\.\d+$', '', p) for p in members.get(g, [])})
    print(f"  the same id means the same family in Results_May21 for {same} of "
          f"{len(core_members)} -- this is the error")

    # ------------------------------------------------ recompute, with the id actually resolved
    pan = pd.read_csv(PAN)
    rows, fixed, kept, empty = [], 0, 0, []
    for _, f in pan.iterrows():
        if str(f.unit).startswith("OG") and f.OG in core_members:
            ids = [p for p in core_members[f.OG] if p in seqs]
            ogb = f"{f.OG} (core-222 list)"
            fixed += 1
        else:
            ogb = f.OG
            ids = [p for p in members.get(ogb, []) if p in seqs]
            kept += 1
        vals = [v for v in (per_protein_cost(seqs[p], aa) for p in ids) if v is not None]
        if not vals:
            empty.append(f.unit)
            continue
        m = np.mean(vals, axis=0)
        rows.append(dict(OG_canonical=f.OG, OG_runB=ogb, unit=f.unit, pathway=f.pathway,
                         module=f.module, CR=f.CR, MR=f.MR, n_proteins=len(vals),
                         **{k: m[i] for i, k in enumerate(METRICS)}))
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "cost_by_OG_351_FIXED.csv", index=False, float_format="%.6g")
    print(f"  {fixed} families re-pointed to a different orthogroup, {kept} unchanged, "
          f"{len(pan) - len(t)} empty {empty if empty else ''}")
    print(f"wrote cost_by_OG_351_FIXED.csv: {len(t)} families, {t.n_proteins.sum():,} proteins")

    # ------------------------------------------------ how wrong was the old table?
    old = pd.read_csv(HERE / "cost_by_OG_351.csv")
    j = t.merge(old[["OG", "Cyano averaged cost"]], left_on="OG_canonical", right_on="OG",
                suffixes=("_fix", "_old"))
    core = j.OG_canonical.isin(core_members) & j.unit.astype(str).str.startswith("OG")
    print("\ncost agreement, fixed vs the table that has been driving Fig 6:")
    for lab, sel in (("core-222 families", core), ("all other families", ~core)):
        q = j[sel]
        r = spearmanr(q["Cyano averaged cost_fix"], q["Cyano averaged cost_old"])
        print(f"  {lab:<20s} n={len(q):>3d}  rho={r.statistic:+.4f}")

    # ------------------------------------------------ headline numbers, before and after
    PETC5 = ["Photosystem II", "Photosystem I", "Cytochrome b6f", "ATP synthase",
             "NADH dehydrogenase (NDH-1)"]
    ANT2 = ["Light-harvesting antenna / phycobiliproteins", "Phycobilisome"]
    scopes = [("whole panel", lambda d: d.index == d.index),
              ("non-photosynthetic", lambda d: ~d.pathway.isin(PETC5 + ANT2)),
              ("PETC, 5 modules", lambda d: d.pathway.isin(PETC5)),
              ("antenna + PBS", lambda d: d.pathway.isin(ANT2))]
    for name, table in (("BEFORE (wrong ids)", old), ("AFTER  (fixed ids)", t)):
        print(f"\n{name}: Cyano averaged cost x CR")
        for lab, sel in scopes:
            q = table[sel(table)]
            r = spearmanr(q["Cyano averaged cost"], q.CR)
            st = "***" if r.pvalue < 1e-3 else "**" if r.pvalue < 1e-2 else \
                 "*" if r.pvalue < 5e-2 else "n.s."
            print(f"  {lab:<22s} n={len(q):>3d}  rho={r.statistic:+.3f}  "
                  f"P={r.pvalue:<9.3g} {st}")
    print("\nAFTER: every metric, whole panel")
    for m in METRICS:
        r = spearmanr(t[m], t.CR)
        st = "***" if r.pvalue < 1e-3 else "**" if r.pvalue < 1e-2 else \
             "*" if r.pvalue < 5e-2 else "n.s."
        print(f"  {m:<30s} rho={r.statistic:+.4f}  P={r.pvalue:<10.3g} {st}")


if __name__ == "__main__":
    main()
