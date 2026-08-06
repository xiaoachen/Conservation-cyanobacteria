#!/usr/bin/env python3
"""Build the R6 master table: biosynthetic cost and conservation on the same protein set.

Why this exists
---------------
Every earlier version of the cost table paired a family's conservation ratio with a cost averaged
over some *other* set of proteins, and the mismatch was invisible because the family name, CR and
pathway all looked right.  Two separate mechanisms caused it:

1.  `26.orthofinder/single_copy_core_OGs_222.tsv` was written on 2026-05-25 from the MCL
    intermediate `Results_May21/WorkingDirectory/clusters_OrthoFinder_I1.2.txt_id_pairs.txt`
    (2026-05-21 17:33), four days before the run finished and wrote
    `Results_May21/Orthogroups/Orthogroups.tsv` (2026-05-29 07:59).  It is one OrthoFinder run --
    started 2026-05-21 17:29:55, finished after 657,013 s -- but MCL and the final output number
    the clusters differently.  The families are identical (222/222 map one-to-one, purity 1.000,
    identical member counts) yet no id is shared.  `compute_cost_351.py` looked the MCL ids up in
    the final table and got a different family for 194 of 351 rows.
2.  `build_whole_panorama.py` de-duplicates on the orthogroup id, so wherever two analysed units
    landed in one orthogroup the second was dropped: 32 units lost, including apcA (shares
    OG0000035 with apcB), psbC (shares OG0000198 with its homologue isiA), cpcB and pecB, plus
    the 28 core families whose MCL id collided with an unrelated final id.

The fix is to stop routing through orthogroup ids altogether.  Each `*_mrcr_results.csv` is one
analysed unit and lists exactly the proteins its CR was measured on; cost is averaged over that
same list.  Conservation and cost then describe one protein set by construction, no lookup can
go wrong, and nothing is dropped.

Inputs (read-only)
  28.mrcr-20250602/??.*/Result/*_mrcr_results.csv          CR + member list, canonical pipeline
  35.metabolism/{Nucleotide...,Lipid_Metabolism}/Result/    same
  34.KaiABC/0*/Result/ , 27.allophycocyanin/Result/         same
  26.orthofinder/protein_sequences_folder/*.pep.faa         36 proteomes, 162,079 sequences
  38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv amino-acid cost table (iJN678, NH4)
  40.whole_CR-MR/panorama_families.csv                      the family set and CR
  26.orthofinder/.../Results_May21/Orthogroups/Orthogroups.tsv  orthogroup audit only
  38.protein_cost/.../background_proteins_costs.tsv         per-protein cost cross-check
Output
  Figure6/cost_master.csv        one row per analysed unit, the panorama 351
  Figure6/cost_master_proteins.csv   one row per protein, for within-family work
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/home/yangyicheng/27.evolution")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "28.mrcr-20250602"))
from plot_mr_cr_jointplot import load_mr_cr_from_dir  # noqa: E402

AA_COST = ROOT / "38.protein_cost/Cyane_cost_cal/outputs/cyano_aa_costs.tsv"
PEP = ROOT / "26.orthofinder/protein_sequences_folder"
PAN = ROOT / "40.whole_CR-MR/panorama_families.csv"
OGT = PEP / "OrthoFinder/Results_May21/Orthogroups/Orthogroups.tsv"
VAL = ROOT / "38.protein_cost/Antenna_protein/outputs/background_proteins_costs.tsv"

METRICS = ["Cyano averaged cost", "Growth-coupled photon cost", "Inorganic carbon cost",
           "ATP turnover demand", "NADPH turnover demand", "N assimilation demand", "Weight"]
STD = set("ACDEFGHIKLMNPQRSTVWY")

# pathway label per source module, copied verbatim from build_whole_panorama.py so that R6 and
# R1-R5 name the same groups the same way
PATHWAY = {
    ("28.mrcr-20250602", "01.singlecopy_gene"): "Single-copy core proteome",
    ("28.mrcr-20250602", "02.Photosystem_II"): "Photosystem II",
    ("28.mrcr-20250602", "03.Photosystem_I"): "Photosystem I",
    ("28.mrcr-20250602", "04.CBB_cycle"): "Calvin-Benson-Bassham cycle",
    ("28.mrcr-20250602", "05.Carbon_concentrating_mechanism"): "Carbon-concentrating mechanism",
    ("28.mrcr-20250602", "06.ATP_synthase"): "ATP synthase",
    ("28.mrcr-20250602", "07.Cytochrome_b6f"): "Cytochrome b6f",
    ("28.mrcr-20250602", "08.Photoprotection"): "Photoprotection",
    ("28.mrcr-20250602", "09.NDH"): "NADH dehydrogenase (NDH-1)",
    ("28.mrcr-20250602", "10.Phycobilisome"): "Phycobilisome",
    ("28.mrcr-20250602", "11.Pigment_biosynthesis"): "Pigment biosynthesis",
    ("35.metabolism", "Nucleotide_Carbohydrate_Amino_Acid_Energy_Metabolism"):
        "Nucleotide/carbohydrate/amino-acid/energy metabolism",
    ("35.metabolism", "Lipid_Metabolism"): "Lipid metabolism",
    ("34.KaiABC", "01.KaiABC"): "Circadian oscillator (KaiABC)",
    ("34.KaiABC", "02.clock-output"): "Circadian clock output",
    ("34.KaiABC", "03.PSII_D1"): "PSII repair protease (FtsH)",
    ("27.allophycocyanin", "Result"): "Light-harvesting antenna / phycobiliproteins",
}
PRECEDENCE = {"Single-copy core proteome": 100}


def find_result_dirs():
    out = []
    for mod in sorted(glob.glob(str(ROOT / "28.mrcr-20250602" / "??.*"))):
        rd = os.path.join(mod, "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("28.mrcr-20250602", os.path.basename(mod), rd))
    for mod in ("Nucleotide_Carbohydrate_Amino_Acid_Energy_Metabolism", "Lipid_Metabolism"):
        rd = str(ROOT / "35.metabolism" / mod / "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("35.metabolism", mod, rd))
    for mod in sorted(glob.glob(str(ROOT / "34.KaiABC" / "0*"))):
        rd = os.path.join(mod, "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("34.KaiABC", os.path.basename(mod), rd))
    rd = str(ROOT / "27.allophycocyanin" / "Result")
    if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
        out.append(("27.allophycocyanin", "Result", rd))
    return out


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


def per_protein_cost(seq, aa):
    counts = defaultdict(int)
    n = 0
    for c in seq:
        if c in STD:
            counts[c] += 1
            n += 1
    if n == 0:
        return None, 0
    tot = np.zeros(len(METRICS))
    for c, k in counts.items():
        tot += aa.loc[c].values * k
    return tot / n, n


def unit_members(csv_path):
    """Every protein the unit's CR was measured on: the reference plus all valid targets.

    Same rows the canonical aggregation keeps -- the orthogroup-median pseudo-rows and the
    entries with no usable MR/CR are dropped, so the cost average covers neither more nor
    fewer proteins than the conservation average.
    """
    df = pd.read_csv(csv_path)
    if not {"MR", "CR"}.issubset(df.columns):
        return []
    tgt = df.get("target", pd.Series([None] * len(df))).astype(str)
    tfile = df.get("target_file", pd.Series([None] * len(df))).astype(str)
    bad = (tgt.eq("__ORTHOGROUP_MEDIAN__") | tgt.eq("NA")
           | tfile.str.contains("__ORTHOGROUP_MEDIAN__", na=False))
    df = df[~bad].copy()
    df["MR"] = pd.to_numeric(df.MR, errors="coerce")
    df["CR"] = pd.to_numeric(df.CR, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MR", "CR"])
    ids = set(df.target.astype(str)) | set(df.ref.astype(str))
    return sorted(re.sub(r"\.\d+$", "", str(i)) for i in ids if str(i).startswith("WP_"))


def main():
    aa = pd.read_csv(AA_COST, sep="\t")
    aa = aa[aa.aa.isin(STD)].set_index("aa")[METRICS]
    seqs = read_proteomes()
    print(f"[1/5] {len(seqs):,} sequences from {len(list(PEP.glob('*.faa')))} proteomes; "
          f"amino-acid cost table = {AA_COST.name} (iJN678, NH4)")

    # ---------------------------------------------------------------- validation gate
    val = pd.read_csv(VAL, sep="\t").dropna(subset=["Cyano averaged cost_mean_per_residue"])
    chk, hit = [], 0
    for _, r in val.sample(min(600, len(val)), random_state=0).iterrows():
        s = seqs.get(re.sub(r"\.\d+$", "", str(r.target_wp)))
        if s is None:
            continue
        v, _ = per_protein_cost(s, aa)
        if v is None:
            continue
        hit += 1
        chk.append([v[i] - r[f"{m}_mean_per_residue"] for i, m in enumerate(METRICS)])
    worst = np.abs(np.array(chk)).max()
    print(f"[2/5] reproduces the published per-protein cost on {hit} proteins, "
          f"largest |difference| = {worst:.3e}")
    if worst > 1e-6:
        raise SystemExit("STOP: cannot reproduce the existing per-protein values.")

    # ---------------------------------------------------------------- one row per analysed unit
    rows, prot_rows, no_seq = [], [], Counter()
    for tree, module, rd in find_result_dirs():
        agg = load_mr_cr_from_dir(rd)                    # canonical CR/MR, one row per unit
        pathway = PATHWAY.get((tree, module), module)
        for _, r in agg.iterrows():
            unit = r["OG"]
            ids = unit_members(os.path.join(rd, f"{unit}_mrcr_results.csv"))
            vals, used = [], []
            for p in ids:
                s = seqs.get(p)
                if s is None:
                    no_seq[unit] += 1
                    continue
                v, n = per_protein_cost(s, aa)
                if v is None:
                    continue
                vals.append(v)
                used.append(p)
                prot_rows.append(dict(unit=unit, tree=tree, pathway=pathway, wp=p, length=n,
                                      **{m: v[i] for i, m in enumerate(METRICS)}))
            if not vals:
                continue
            m = np.mean(vals, axis=0)
            rows.append(dict(tree=tree, module=module, pathway=pathway, unit=unit,
                             CR=r["CR"], MR=r["MR"], n_cr=len(ids), n_cost=len(used),
                             **{k: m[i] for i, k in enumerate(METRICS)}))
    master = pd.DataFrame(rows)
    print(f"[3/5] {len(master)} analysed units costed "
          f"({master.n_cost.sum():,} proteins; {sum(no_seq.values())} accessions had no sequence)")

    # ---------------------------------------------------------------- de-duplicate by unit name
    # the same gene is analysed under two trees in a few places (psbC in both the PSII and the
    # metabolism sweep, apcA in both the phycobilisome and the allophycocyanin sweep).  Keep the
    # more specific pathway, then the version resting on more proteins -- the rule
    # build_whole_panorama.py uses, applied to the unit rather than to the orthogroup id.
    # R6 uses exactly the family set the rest of the paper uses: the 351 rows of
    # panorama_families.csv, matched on (unit, tree) so that the conservation ratio is the same
    # number R1-R5 print rather than merely a close one.  Restricting rather than extending is a
    # deliberate choice: the 32 further units this pipeline can cost are excluded so that R6's n
    # matches every other section.  What that costs is recorded in the caveat printed below.
    pan = pd.read_csv(PAN)
    pan_pick = set(map(tuple, pan[["unit", "tree"]].values))
    keep = [(u, tr) in pan_pick for u, tr in zip(master.unit, master.tree)]
    t = master[keep].drop_duplicates("unit", keep="first").reset_index(drop=True)
    print(f"[4/5] restricted to the panorama family set: {len(t)} of {len(master)} analysed units")
    lost = sorted(set(master.unit) - set(t.unit))
    print(f"      excluded {len(lost)} units panorama does not carry: {lost}")

    # ---------------------------------------------------------------- cross-check against R1-R5
    j = t.merge(pan[["unit", "CR", "pathway"]], on="unit", suffixes=("", "_pan"))
    d = (j.CR - j.CR_pan).abs()
    print(f"[5/5] CR cross-check against panorama_families.csv: {len(j)} of {len(pan)} rows "
          f"matched, identical to 1e-9 for {(d < 1e-9).sum()}, max |difference| {d.max():.2e}")
    missing = sorted(set(pan.unit) - set(t.unit))
    if missing:
        print(f"      WARNING: {len(missing)} panorama rows could not be costed: {missing}")

    # ---------------------------------------------------------------- the caveat, on the record
    # The 351 rows are analysed units, not distinct orthogroups.  Mapping each unit's proteins
    # back to Results_May21 shows 28 orthogroups carrying two rows each -- one named by gene
    # symbol, one by the earlier run's orthogroup id -- so 351 units cover 323 orthogroups.  That
    # is inherited from panorama and kept deliberately so that R6 counts families the way the
    # other sections do; it must be stated in Methods.
    og = pd.read_csv(OGT, sep="\t", dtype=str)
    wp2og = {}
    for _, r in og.iterrows():
        for cell in r.iloc[1:]:
            if isinstance(cell, str) and cell.strip():
                for x in cell.split(", "):
                    wp2og[re.sub(r"\.\d+$", "", x.strip())] = r.iloc[0]
    pp = pd.DataFrame(prot_rows)
    pp = pp[pp.unit.isin(t.unit)].copy()
    pp["ogF"] = pp.wp.map(wp2og)
    per_unit = pp.dropna(subset=["ogF"]).groupby("unit").ogF.agg(
        lambda s: s.value_counts().index[0])
    n_og = per_unit.nunique()
    dup = per_unit.value_counts()
    dup = dup[dup > 1]
    print(f"      caveat: these {len(t)} units cover {n_og} distinct orthogroups; "
          f"{len(dup)} orthogroups carry {int(dup.sum())} units between them")
    t["orthogroup_final"] = t.unit.map(per_unit)
    t["orthogroup_shared_with"] = [
        ", ".join(sorted(set(per_unit[per_unit == g].index) - {u})) or ""
        for u, g in zip(t.unit, t.orthogroup_final)]

    t.to_csv(HERE / "cost_master.csv", index=False, float_format="%.6g")
    pp.drop(columns="ogF").to_csv(HERE / "cost_master_proteins.csv", index=False,
                                  float_format="%.6g")
    print(f"\nwrote cost_master.csv ({len(t)} families, {t.n_cost.sum():,} protein slots, "
          f"{pp.wp.nunique():,} distinct proteins) and cost_master_proteins.csv")
    print(t.pathway.value_counts().to_string())


if __name__ == "__main__":
    main()
