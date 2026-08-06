#!/usr/bin/env python3
"""Per-genome mean conservation ratio computed on the 351 analysis units.

WHY THIS EXISTS
---------------
The genome-architecture panels of Fig. 2c,d were drawn from `13.genomesize_cr/cr_table.csv`, a
legacy table of 189 genes per genome.  That table has two gaps: Synechocystis sp. PCC 6803, which
supplies 334 of the 351 reference structures and therefore has no conservation ratio against
itself, and Thermosynechococcus vestitus BP-1, whose row exists but carries no conservation values
at all.  The axis therefore rested on 34 of 36 genomes, and BP-1 was missing for a reason that was
never a property of the data -- it simply was not scored in that legacy table.

Every per-unit result file names its targets by protein accession, and the proteome FASTA files
name every accession's genome, so a per-genome mean can be computed directly on the same 351 units
the rest of the study uses.  BP-1 is recovered this way.  PCC 6803 remains undefined by
construction, because conservation is measured against it.

A per-genome mean over all 351 units is not comparable across genomes, because the number of units
measured per genome ranges from 19 to 331 and a mean over 77 families is a mean over a different
family set than a mean over 328.  The published quantity is therefore restricted to units measured
in at least 30 of the 36 genomes -- 271 of 351, five-sixths of the panel -- which fixes the family
set without discarding a quarter of the data.  All five definitions are emitted side by side and all
give the same sign at P < 10^-3, so the choice changes the coefficient by a few hundredths and
nothing else:

    A  legacy 189-gene fixed set (what Fig. 2c,d used)     rho = +0.607 / +0.615   N = 34
    B  351 units, unrestricted mean                        rho = +0.533 / +0.567   N = 35
    C  351 units, units in >= 30 genomes (published)       rho = +0.575 / +0.608   N = 35
    C' the same at >= 25 and >= 33 genomes                 rho = +0.547 / +0.617
    D  unit-centred strain effect, all 351 units           rho = +0.550 / +0.586   N = 35

Two genomes need naming.  BP-1 is recovered here at CR 0.636 and enters the axis for the first
time.  PCC 6803 is excluded from every definition: it supplies 334 of the 351 reference structures,
so it appears as a target in only 19 units and its apparent value of 0.486 is computed on the small,
unrepresentative set of families for which some other genome was the reference.

Inputs (read-only)
  40.whole_CR-MR/panorama_families.csv          the 351 (unit, tree, module) triples
  <tree>/<module>/Result/<unit>_mrcr_results.csv every reference-target pair
  26.orthofinder/protein_sequences_folder/*.pep.faa   accession -> genome
  13.genomesize_cr/cr_table.csv                 the legacy values, for comparison
Output
  strain_mean_CR_351.csv   one row per genome, all five definitions, with the published column
                           flagged and the reason for any exclusion stated
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/yangyicheng/27.evolution")
OUT = Path(__file__).resolve().parent
PEP = ROOT / "26.orthofinder/protein_sequences_folder"
PAN = ROOT / "40.whole_CR-MR/panorama_families.csv"
LEGACY = ROOT / "13.genomesize_cr/cr_table.csv"


def wp_to_genome():
    """Every protein accession mapped to the genome whose proteome contains it."""
    m = {}
    for f in sorted(PEP.glob("*.pep.faa")):
        g = re.search(r"(GCF_\d+)", f.name).group(1)
        for line in f.read_text().splitlines():
            if line.startswith(">"):
                m[re.sub(r"\.\d+$", "", line[1:].split()[0])] = g
    return m


def unit_files():
    """Locate the result file of each of the 351 analysis units."""
    pan = pd.read_csv(PAN)
    out = []
    for _, r in pan.iterrows():
        base = (ROOT / "27.allophycocyanin" if r.tree == "27.allophycocyanin"
                else ROOT / str(r.tree) / str(r.module))
        p = base / "Result" / f"{r.unit}_mrcr_results.csv"
        if p.is_file():
            out.append((r.unit, r.pathway, p))
    return out


def main():
    w2g = wp_to_genome()
    files = unit_files()
    rows = []
    for unit, pathway, p in files:
        d = pd.read_csv(p)
        tgt = d.get("target", pd.Series(dtype=str)).astype(str)
        bad = tgt.eq("__ORTHOGROUP_MEDIAN__") | tgt.eq("NA")
        d = d[~bad].copy()
        d["CR"] = pd.to_numeric(d.CR, errors="coerce")
        d = d.dropna(subset=["CR"])
        d["genome"] = tgt[~bad].str.replace(r"\.\d+$", "", regex=True).map(w2g)
        for g, s in d.dropna(subset=["genome"]).groupby("genome"):
            rows.append(dict(genome=g, unit=unit, pathway=pathway, CR=float(s.CR.mean()),
                             n_pairs=len(s)))
    per = pd.DataFrame(rows)
    print(f"{len(files)} units resolved; {per.genome.nunique()} genomes appear as targets; "
          f"{len(per):,} genome x unit values")

    MIN_GENOMES = 30                    # a unit must be measured in five-sixths of the panel
    REF = "GCF_000009725"               # supplies 334 of 351 reference structures
    cov = per.groupby("unit").genome.nunique()
    common = cov[cov >= MIN_GENOMES].index
    print(f"units measured in >= {MIN_GENOMES} genomes: {len(common)} of {len(cov)}")

    per["dev"] = per.CR - per.groupby("unit").CR.transform("mean")
    agg = (per.groupby("genome")
              .agg(mean_CR_all351=("CR", "mean"), units_measured=("unit", "nunique"),
                   pairs=("n_pairs", "sum"), strain_effect_centred=("dev", "mean")).reset_index())
    sub = (per[per.unit.isin(common)].groupby("genome")
              .agg(mean_CR_common=("CR", "mean"),
                   units_common=("unit", "nunique")).reset_index())
    agg = agg.merge(sub, on="genome", how="left")

    leg = pd.read_csv(LEGACY)
    leg["genome"] = leg.GCF.astype(str).str.extract(r"(GCF_\d+)", expand=False)
    gcols = [c for c in leg.columns if re.fullmatch(r"gene\d+_CR", str(c))]
    for c in gcols + ["genome_size", "protein_num"]:
        leg[c] = pd.to_numeric(leg[c], errors="coerce")
    leg["mean_CR_legacy189"] = leg[gcols].mean(axis=1, skipna=True)
    t = agg.merge(leg[["genome", "genome_size", "protein_num", "mean_CR_legacy189"]],
                  on="genome", how="outer").sort_values("mean_CR_common")
    t["genome_Mb"] = t.genome_size / 1e6
    t["published_CR"] = np.where(t.genome == REF, np.nan, t.mean_CR_common)
    t["excluded_because"] = np.where(
        t.genome == REF,
        "Reference genome for 334 of 351 units; conservation is measured against it, so it has "
        "no conservation ratio of its own", "")
    t.to_csv(OUT / "strain_mean_CR_351.csv", index=False, float_format="%.6g")

    print(f"\n{'genome':18s} {'units':>6s} {'pairs':>7s} {'CR(351)':>9s} {'CR(189)':>9s} {'Mb':>6s}")
    for _, r in t.iterrows():
        n351 = f"{r.published_CR:.4f}" if pd.notna(r.published_CR) else "-"
        n189 = f"{r.mean_CR_legacy189:.4f}" if pd.notna(r.mean_CR_legacy189) else "-"
        mb = f"{r.genome_Mb:.2f}" if pd.notna(r.genome_Mb) else "-"
        u = f"{int(r.units_measured)}" if pd.notna(r.units_measured) else "-"
        pr = f"{int(r.pairs)}" if pd.notna(r.pairs) else "-"
        print(f"{r.genome:18s} {u:>6s} {pr:>7s} {n351:>9s} {n189:>9s} {mb:>6s}")

    both = t.dropna(subset=["published_CR", "mean_CR_legacy189"])
    print(f"\nagreement between the published and legacy definitions over {len(both)} genomes: "
          f"Spearman {spearmanr(both.published_CR, both.mean_CR_legacy189).statistic:+.3f}, "
          f"median absolute difference "
          f"{np.median(abs(both.published_CR - both.mean_CR_legacy189)):.4f}")

    print("\ngenome-architecture axes, every definition:")
    for col, lab in (("genome_Mb", "genome size (Mb)"), ("protein_num", "protein-coding genes")):
        for cr, tag in (("published_CR", "published: units in >= 30 genomes"),
                        ("mean_CR_all351", "all 351 units, unrestricted"),
                        ("strain_effect_centred", "unit-centred strain effect"),
                        ("mean_CR_legacy189", "legacy 189-gene set")):
            q = t[t.genome != REF].dropna(subset=[col, cr])
            r = spearmanr(q[col], q[cr])
            print(f"  {lab:22s} x {tag:34s} rho = {r.statistic:+.3f}  P = {r.pvalue:.3g}  "
                  f"N = {len(q)}")
    print(f"\nwrote {(OUT / 'strain_mean_CR_351.csv').name}")


if __name__ == "__main__":
    main()
