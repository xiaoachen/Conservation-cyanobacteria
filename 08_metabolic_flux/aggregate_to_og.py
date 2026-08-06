#!/usr/bin/env python3
"""Aggregate flux_native_merged.csv from (Strain, Model, OG, Reaction, Gene)
granularity down to:

  flux_native_og_per_model.csv     one row per (Strain_GCF, Model, OG)
                                   — used for per-model Spearman against CR
  flux_native_og_consensus.csv     one row per OG (median / mean across all models)
                                   — used for the single rigorous global statistic

For each (Strain, Model, OG), we collapse the multiple gene/reaction rows by:
  Flux_Amplitude_max     max over all (rxn, gene) rows (representative peak flux)
  Flux_Amplitude_sum     sum across all reactions (total flux through this OG)
  Flux_Amplitude_median  median across rows
  Flux_Light_max, Flux_Dark_max  for the corresponding light/dark fluxes
  n_reactions, n_genes
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/yangyicheng/27.evolution/39.flux_CR")
INPUT = ROOT / "04.merged/flux_native_merged.csv"
OUT_PERMODEL = ROOT / "04.merged/flux_native_og_per_model.csv"
OUT_CONSENSUS = ROOT / "04.merged/flux_native_og_consensus.csv"


def main():
    df = pd.read_csv(INPUT)
    for col in ("Flux_Light", "Flux_Dark", "Flux_Diff", "Flux_Amplitude", "CR", "MR"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Flux_Amplitude", "CR"])
    df = df[(df["CR"] >= 0) & (df["CR"] <= 1)]

    # Per-(Strain, Model, OG) collapse
    grp = df.groupby(["Strain_GCF", "Strain_TaxID", "Organism", "Model", "OG"], as_index=False)
    per_model = grp.agg(
        Flux_Light_max=("Flux_Light", "max"),
        Flux_Dark_max=("Flux_Dark", "max"),
        Flux_Amp_max=("Flux_Amplitude", "max"),
        Flux_Amp_sum=("Flux_Amplitude", "sum"),
        Flux_Amp_median=("Flux_Amplitude", "median"),
        n_reactions=("Reaction_ID", "nunique"),
        n_genes=("Gene", "nunique"),
        Reactions=("Reaction_ID", lambda x: ";".join(sorted(set(x)))),
    )
    # Bring in CR/MR/Category from any row (constant per OG)
    cr_cols = df.drop_duplicates("OG")[["OG", "MR", "CR", "Category", "Mean_Ref_Length", "N_Strains_in_OG"]]
    per_model = per_model.merge(cr_cols, on="OG", how="left")
    per_model.to_csv(OUT_PERMODEL, index=False)
    print(f"Wrote {len(per_model)} rows -> {OUT_PERMODEL}", file=sys.stderr)

    # Per-OG consensus: aggregate across models within a strain first, then across strains
    # We have to be careful — same OG can appear in multiple models; we want a "real"
    # per-OG record that summarizes its flux across the metabolic-modeling literature.
    grp_og = per_model.groupby(["OG", "MR", "CR", "Category", "Mean_Ref_Length", "N_Strains_in_OG"], as_index=False, dropna=False)
    consensus = grp_og.agg(
        Flux_Amp_median=("Flux_Amp_max", "median"),
        Flux_Amp_mean_log2=("Flux_Amp_max", lambda x: np.mean(np.log2(x[x > 0]))) ,
        Flux_Amp_min=("Flux_Amp_max", "min"),
        Flux_Amp_max=("Flux_Amp_max", "max"),
        n_models=("Model", "nunique"),
        n_strains=("Strain_GCF", "nunique"),
        Strains=("Strain_GCF", lambda x: ";".join(sorted(set(x)))),
        Models=("Model", lambda x: ";".join(sorted(set(x)))),
    )
    consensus.to_csv(OUT_CONSENSUS, index=False)
    print(f"Wrote {len(consensus)} rows -> {OUT_CONSENSUS}", file=sys.stderr)

    # Quick stats
    from scipy.stats import spearmanr
    print()
    print("==== Per-(Strain, Model) at OG-level (max flux per OG) ====")
    for (g, m), sub in per_model.groupby(["Strain_GCF", "Model"]):
        sub = sub[sub["Flux_Amp_max"] > 0]
        if len(sub) < 5:
            continue
        rho, p = spearmanr(np.log2(sub["Flux_Amp_max"]), sub["CR"])
        org = sub["Organism"].iloc[0]
        print(f"  {g} {m:25s}  N={len(sub):4d}  ρ={rho:+.3f}  p={p:.3e}   {org}")

    pm = per_model[per_model["Flux_Amp_max"] > 0]
    rho, p = spearmanr(np.log2(pm["Flux_Amp_max"]), pm["CR"])
    print(f"  POOLED-per-(Strain, Model, OG)                 N={len(pm):4d}  ρ={rho:+.3f}  p={p:.3e}")

    print()
    print("==== Per-OG consensus (median across all (strain, model) instances) ====")
    cs = consensus[consensus["Flux_Amp_median"] > 0]
    rho, p = spearmanr(np.log2(cs["Flux_Amp_median"]), cs["CR"])
    print(f"  All OGs              N={len(cs):4d}  ρ={rho:+.3f}  p={p:.3e}")
    cs3 = cs[cs["n_models"] >= 3]
    rho, p = spearmanr(np.log2(cs3["Flux_Amp_median"]), cs3["CR"])
    print(f"  OGs in ≥3 models    N={len(cs3):4d}  ρ={rho:+.3f}  p={p:.3e}")


if __name__ == "__main__":
    main()
