#!/usr/bin/env python3
"""Aggregate existing per-target core/surface CR (Result_Structure/*_mrcr_results.csv) to the
351 panorama families. complete_CR = panorama whole-protein CR. Reports coverage per category."""
import os, glob
import pandas as pd, numpy as np

ROOT = "/home/yangyicheng/27.evolution"
OUTDIR = "/home/yangyicheng/27.evolution/40.whole_CR-MR"
pan = pd.read_csv(f"{OUTDIR}/panorama_families.csv")


def base_dir(tree, module):
    if tree == "27.allophycocyanin":
        return os.path.join(ROOT, "27.allophycocyanin")
    return os.path.join(ROOT, tree, str(module))


def agg_struct(tree, module, unit):
    """mean core_CR/surface_CR over targets from Result_Structure/<unit>_mrcr_results.csv."""
    p = os.path.join(base_dir(tree, module), "Result_Structure", f"{unit}_mrcr_results.csv")
    if not os.path.isfile(p):
        return np.nan, np.nan
    try:
        d = pd.read_csv(p)
    except Exception:
        return np.nan, np.nan
    d = d[~d["target"].astype(str).str.startswith("__")]
    for c in ("core_CR", "surface_CR"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d["core_CR"].mean(), d["surface_CR"].mean()


rows = []
for _, r in pan.iterrows():
    core, surf = agg_struct(r["tree"], r["module"], r["unit"])
    rows.append(dict(family=r["family"], unit=r["unit"], tree=r["tree"], pathway=r["pathway"],
                     complete_CR=r["CR"], core_CR=core, surface_CR=surf))
out = pd.DataFrame(rows)
out.to_csv(f"{OUTDIR}/panorama_core_surface_351.csv", index=False)

n = len(out)
print(f"families: {n}")
print(f"complete_CR present : {out.complete_CR.notna().sum()}")
print(f"core/surface present: {out.core_CR.notna().sum()}  (missing {out.core_CR.isna().sum()})")
print("missing core/surface by tree:")
print(out[out.core_CR.isna()].groupby('tree').size().to_string())
print()
sub = out.dropna(subset=['core_CR', 'surface_CR'])
from scipy.stats import spearmanr
rho, p = spearmanr(sub.core_CR, sub.surface_CR)
print(f"[351-set] core_CR vs surface_CR: N={len(sub)}, rho={rho:.2f}, P={p:.1e}")
print(f"mean complete={out.complete_CR.mean():.3f}  core={sub.core_CR.mean():.3f}  surface={sub.surface_CR.mean():.3f}")
