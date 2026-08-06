#!/usr/bin/env python3
"""
Whole-project CR/MR panorama.

Collects every analyzed protein unit (one *_mrcr_results.csv = one gene/OG) across
the four AF3-covered trees, reuses the already-computed MR/CR values (same filtering
as the canonical 28.mrcr plotting pipeline), maps each unit to a 26.orthofinder
orthogroup (gene family), deduplicates to family level, and renders a single
panoramic MR vs CR joint plot. Also tallies genes, gene families, AF3 structures and
pathway breakdown.

Run inside the `mrcr` conda env (seaborn/scipy/pandas available there).
"""
import os
import sys
import glob
import collections

import pandas as pd
import numpy as np

ROOT = "/home/yangyicheng/27.evolution"
OUT = os.path.join(ROOT, "40.whole_CR-MR")
ORTHO_TSV = os.path.join(
    ROOT,
    "26.orthofinder/protein_sequences_folder/OrthoFinder/Results_May21/Orthogroups/Orthogroups.tsv",
)
SINGLECOPY_TSV = os.path.join(ROOT, "26.orthofinder/single_copy_core_OGs_222.tsv")

# Reuse the canonical per-file MR/CR aggregation (same pLDDT/length filtering)
sys.path.insert(0, os.path.join(ROOT, "28.mrcr-20250602"))
from plot_mr_cr_jointplot import load_mr_cr_from_dir, plot_joint  # noqa: E402


# ---- pathway labels per source module ----------------------------------------
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

# precedence for dedup: lower number wins (keep the most specific pathway).
# single-copy core is generic -> kept only when nothing more specific exists.
PRECEDENCE = {"Single-copy core proteome": 100}


def build_wp2og():
    """bare WP accession -> OG id, from the full OrthoFinder assignment."""
    wp2og = {}
    with open(ORTHO_TSV) as fh:
        next(fh)  # header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            og = parts[0]
            for cell in parts[1:]:
                cell = cell.strip()
                if not cell:
                    continue
                for g in cell.split(", "):
                    g = g.strip()
                    if g:
                        wp2og[g] = og
    return wp2og


def core222_set():
    ogs = set()
    with open(SINGLECOPY_TSV) as fh:
        next(fh)
        for line in fh:
            ogs.add(line.split("\t", 1)[0])
    return ogs


def find_result_dirs():
    """(tree, module, result_dir) for every Result/ holding *_mrcr_results.csv."""
    out = []
    # 28.mrcr functional + core modules
    for mod in sorted(glob.glob(os.path.join(ROOT, "28.mrcr-20250602", "??.*"))):
        rd = os.path.join(mod, "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("28.mrcr-20250602", os.path.basename(mod), rd))
    # 35.metabolism subdirs (top-level Result has no csv)
    for mod in ("Nucleotide_Carbohydrate_Amino_Acid_Energy_Metabolism", "Lipid_Metabolism"):
        rd = os.path.join(ROOT, "35.metabolism", mod, "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("35.metabolism", mod, rd))
    # 34.KaiABC
    for mod in sorted(glob.glob(os.path.join(ROOT, "34.KaiABC", "0*"))):
        rd = os.path.join(mod, "Result")
        if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
            out.append(("34.KaiABC", os.path.basename(mod), rd))
    # 27.allophycocyanin
    rd = os.path.join(ROOT, "27.allophycocyanin", "Result")
    if glob.glob(os.path.join(rd, "*_mrcr_results.csv")):
        out.append(("27.allophycocyanin", "Result", rd))
    return out


def scan_members(unit_dir):
    """Return (member_keys, af3_keys) for a unit folder of WP_*_GCF_* member dirs.
    member_key = (wp_accession, gcf); af3_keys = members carrying a top-level *_model.cif."""
    keys, af3_keys = [], []
    for mdir in glob.glob(os.path.join(unit_dir, "WP_*_GCF_*")):
        if not os.path.isdir(mdir):
            continue
        base = os.path.basename(mdir)
        if "_GCF_" not in base:
            continue
        wp, gcf = base.split("_GCF_", 1)
        key = (wp, "GCF_" + gcf)
        keys.append(key)
        # AF3 ranked model lives in the lowercased job subdir as *_model.cif
        if glob.glob(os.path.join(mdir, "*", "*_model.cif")):
            af3_keys.append(key)
    return keys, af3_keys


def main():
    os.makedirs(OUT, exist_ok=True)
    print("[1/5] building WP->OG map ...")
    wp2og = build_wp2og()
    core222 = core222_set()
    print(f"      {len(wp2og):,} genes mapped across {len(set(wp2og.values())):,} OGs; "
          f"{len(core222)} single-copy core OGs")

    print("[2/5] scanning Result dirs ...")
    rdirs = find_result_dirs()
    rows = []
    global_members = set()      # (wp,gcf) across whole project
    global_af3_jobs = set()     # member dirs carrying a model.cif
    for tree, module, rd in rdirs:
        try:
            agg = load_mr_cr_from_dir(rd)  # one row per *_mrcr_results.csv
        except Exception as e:
            print(f"      [WARN] {tree}/{module}: {e}")
            continue
        unit_root = os.path.dirname(rd)  # sibling-of-Result holds <unit>/ folders
        pathway = PATHWAY.get((tree, module), module)
        for _, r in agg.iterrows():
            unit = r["OG"]  # filename stem: OG id or gene name
            unit_dir = os.path.join(unit_root, unit)
            members, af3_keys = scan_members(unit_dir)
            ogs = [wp2og[wp] for wp, _ in members if wp in wp2og]
            if unit.startswith("OG") and unit in core222:
                og = unit                      # core unit: filename is the OG
            elif ogs:
                og = collections.Counter(ogs).most_common(1)[0][0]
            else:
                og = ""                        # unassigned -> own family by gene name
            family = og if og else f"GENE:{unit}"
            global_members.update(members)
            global_af3_jobs.update(af3_keys)  # only members with an actual model.cif
            rows.append(dict(
                tree=tree, module=module, pathway=pathway, unit=unit,
                OG=og, in_core222=(og in core222), family=family,
                n_members=len(members), n_af3=len(af3_keys),
                MR=r["MR"], CR=r["CR"],
            ))
        print(f"      {tree}/{module}: {len(agg)} units")

    master = pd.DataFrame(rows)
    master.to_csv(os.path.join(OUT, "master_units.csv"), index=False)
    print(f"      total analyzed units: {len(master)}")

    print("[3/5] de-duplicating to family level ...")
    master["prec"] = master["pathway"].map(lambda p: PRECEDENCE.get(p, 0))
    # keep most-specific pathway per family, then larger member count
    fam = (master.sort_values(["prec", "n_members"], ascending=[True, False])
                 .drop_duplicates("family", keep="first")
                 .reset_index(drop=True))
    fam.to_csv(os.path.join(OUT, "panorama_families.csv"), index=False)
    n_overlap = len(master) - len(fam)
    print(f"      unique families: {len(fam)} (collapsed {n_overlap} duplicate units)")

    print("[4/5] rendering panorama joint plot ...")
    plot_joint(fam[["CR", "MR"]].copy(),
               os.path.join(OUT, "whole_CR-MR_joint.png"), dpi=600)

    print("[5/5] writing summary ...")
    n_genes = len(global_members)
    n_wp = len({wp for wp, _ in global_members})
    n_af3 = len(global_af3_jobs)
    fam_assigned = fam[fam["OG"] != ""]
    n_fam = fam["family"].nunique()
    n_fam_og = fam_assigned["OG"].nunique()
    n_fam_core = fam_assigned[fam_assigned["in_core222"]]["OG"].nunique()
    n_fam_noncore = n_fam_og - n_fam_core
    n_fam_unassigned = (fam["OG"] == "").sum()

    pw = (fam.groupby("pathway")
             .agg(families=("family", "nunique"),
                  median_CR=("CR", "median"),
                  median_MR=("MR", "median"))
             .sort_values("families", ascending=False))

    lines = []
    lines.append("WHOLE-PROJECT CR/MR PANORAMA — SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Trees scanned        : 28.mrcr-20250602, 35.metabolism, 34.KaiABC, 27.allophycocyanin")
    lines.append(f"Analyzed units       : {len(master)}  (one *_mrcr_results.csv each)")
    lines.append(f"Unique gene families : {n_fam}")
    lines.append(f"  - mapped to 26.orthofinder OGs : {n_fam_og}")
    lines.append(f"      of which single-copy core 222: {n_fam_core}")
    lines.append(f"      additional (non-core) OGs    : {n_fam_noncore}")
    lines.append(f"  - unassigned (no OrthoFinder OG) : {n_fam_unassigned}")
    lines.append(f"Cyanobacterial genes : {n_genes:,}  (distinct protein members = WP x strain)")
    lines.append(f"  - distinct WP accessions (non-redundant proteins): {n_wp:,}")
    lines.append(f"AlphaFold3 structures: {n_af3:,}  (members with a ranked *_model.cif)")
    lines.append(f"Duplicate units collapsed in panorama: {n_overlap}")
    lines.append("")
    lines.append("Pathway / functional-category breakdown (deduped families):")
    lines.append(pw.to_string())
    summary = "\n".join(lines)
    with open(os.path.join(OUT, "summary.txt"), "w") as fh:
        fh.write(summary + "\n")
    print("\n" + summary)


if __name__ == "__main__":
    main()
