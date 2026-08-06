#!/usr/bin/env python3
"""Buried-residue fraction for every reference structure in the 351-family panel.

WHY
---
R4's load-bearing claim is compositional: reaction-centre subunits are extremely conserved
because they are disproportionately BUILT OUT OF buried structural environments.  The test is
buried fraction versus protein-mean CR, and until now it existed only for the 20 proteins with
residue-level RSA already on disk (Supplementary Fig. S4c, N = 13 photosynthetic
electron-transport proteins, rho = +0.621, P = 0.024).  Thirteen points is the thinnest link in
the paper's main line, against N = 337 on the metabolic-demand side.

The fraction cannot be recovered from what is already stored.  `panorama_core_surface_351.csv`
holds complete_CR, core_CR and surface_CR per family, which suggests solving
    f = (CR_complete - CR_surface) / (CR_core - CR_surface)
but that identity does not hold here: tested against the 19 proteins whose true buried fraction
is known it gives Pearson r = 0.161 and a mean absolute error of 0.113, with atpB off by +0.60
and petE estimated at 1.081.  The three quantities are computed over different residue sets and
aggregated differently, and where core_CR and surface_CR are close (psbA 0.848 versus 0.820) the
denominator collapses.  So the RSA has to be recomputed.

WHAT THIS DOES
--------------
For each of the 351 families: pick the reference AF3 structure by the same rule the binding-site
masks used (prefer Synechocystis PCC 6803 GCF_000009725, then Thermosynechococcus GCF_000011345,
otherwise the longest available model), run ChimeraX `measure sasa` on it, convert to relative
solvent accessibility against the Tien et al. (2013) theoretical maxima, and record the fraction
of residues below RSA 0.25 -- the same cutoff that defines the core/surface split everywhere else
in the paper.

Two fractions are reported.  `f_buried` uses every residue of the reference; `f_buried_p70` is
restricted to residues with pLDDT >= 70, matching the mask applied to all other CR analyses, and
is the one to use.  They are reported separately so the effect of the mask is visible rather than
assumed.

Reuses `28.mrcr-20250602/structure_mr_cr_ChimeraX.py` verbatim (read_cif_structure,
compute_properties_with_chimerax, get_rsa) so this script cannot drift from the pipeline that
produced the core/surface CR values in Fig. 4.

Run in the mrcr environment (ChimeraX 1.7.1, gemmi):
    /home/yangyicheng/mambaforge/envs/mrcr/bin/python compute_buried_fraction_351.py

Output: buried_fraction_351.csv   one row per family, written incrementally so a crash or a
                                  kill loses only the family in flight
        buried_fraction_351.log   per-family progress and failures
"""
import os
import sys
import glob
import shutil
import tempfile
import traceback
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = "/home/yangyicheng/27.evolution"
PANEL = f"{ROOT}/40.whole_CR-MR/panorama_families.csv"
OUT = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(OUT, "buried_fraction_351.csv")
LOG = os.path.join(OUT, "buried_fraction_351.log")

RSA_CORE = 0.25        # buried if RSA < this; the cutoff used throughout the paper
MIN_PLDDT = 70.0       # the confidence mask applied to every other CR analysis

sys.path.insert(0, os.path.join(ROOT, "28.mrcr-20250602"))
import structure_mr_cr_ChimeraX as S      # noqa: E402


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def base_dir(tree, module):
    if tree == "27.allophycocyanin":
        return os.path.join(ROOT, "27.allophycocyanin")
    return os.path.join(ROOT, tree, str(module))


def cif_nres(p):
    try:
        return sum(1 for ln in open(p) if ln.startswith("ATOM") and " CA " in ln)
    except Exception:
        return 0


def pick_ref(unit_dir):
    """Same reference-selection rule as build_upb_masks_351.py."""
    for gcf in ("GCF_000009725", "GCF_000011345"):
        for pat in (f"WP_*_{gcf}/*/seed-1_sample-0/model.cif", f"WP_*_{gcf}/*/*_model.cif"):
            g = glob.glob(os.path.join(unit_dir, pat))
            if g:
                return g[0]
    cands = (glob.glob(os.path.join(unit_dir, "WP_*_GCF_*/*/seed-1_sample-0/model.cif")) or
             glob.glob(os.path.join(unit_dir, "WP_*_GCF_*/*/*_model.cif")))
    return max(cands, key=cif_nres) if cands else None


def sasa_for(cif_path):
    """Per-residue SASA aligned to the residues returned by read_cif_structure."""
    seq, ca, res_ids, plddt = S.read_cif_structure(cif_path)
    tmp = tempfile.mkdtemp(dir=OUT, prefix=".sasa_")
    try:
        _, json_path = S.compute_properties_with_chimerax(cif_path, temp_dir=tmp)
        import json
        props = json.load(open(json_path))
        pmap = {(int(p["seqid"]), str(p.get("icode", "")).strip()):
                float(p.get("sasa", 0.0)) for p in props}
        sasa = np.array([pmap.get((rid[0], str(rid[1]).strip()), np.nan) for rid in res_ids])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return seq, np.asarray(plddt, float), sasa


def main():
    pan = pd.read_csv(PANEL)
    done = set()
    if os.path.exists(CSV):
        done = set(pd.read_csv(CSV).family)
        log(f"resuming: {len(done)} families already in {os.path.basename(CSV)}")
    header_needed = not os.path.exists(CSV)

    log(f"{len(pan)} families in the panel, {len(pan) - len(done)} to do "
        f"(RSA cutoff {RSA_CORE}, pLDDT mask {MIN_PLDDT:g})")

    for i, r in pan.reset_index(drop=True).iterrows():
        if r.family in done:
            continue
        tag = f"{i+1}/{len(pan)} {r.unit}"
        unit_dir = os.path.join(base_dir(r.tree, r.module), str(r.unit))
        ref = pick_ref(unit_dir)
        row = dict(family=r.family, unit=r.unit, tree=r.tree, module=r.module,
                   pathway=r.pathway, ref_path=ref or "", n_res=0, n_res_p70=0,
                   n_buried=0, n_buried_p70=0, f_buried=np.nan, f_buried_p70=np.nan,
                   mean_rsa=np.nan, mean_rsa_p70=np.nan, mean_plddt=np.nan, status="")
        if ref is None:
            row["status"] = "no reference structure"
            log(f"  {tag}: SKIP, no reference structure under {unit_dir}")
        else:
            try:
                seq, plddt, sasa = sasa_for(ref)
                ok = np.isfinite(sasa)
                rsa = S.get_rsa(seq, np.nan_to_num(sasa, nan=0.0))
                p70 = ok & (plddt >= MIN_PLDDT)
                buried = rsa < RSA_CORE
                row.update(
                    n_res=int(ok.sum()), n_res_p70=int(p70.sum()),
                    n_buried=int((buried & ok).sum()), n_buried_p70=int((buried & p70).sum()),
                    f_buried=float((buried & ok).sum() / ok.sum()) if ok.sum() else np.nan,
                    f_buried_p70=float((buried & p70).sum() / p70.sum()) if p70.sum() else np.nan,
                    mean_rsa=float(rsa[ok].mean()) if ok.sum() else np.nan,
                    mean_rsa_p70=float(rsa[p70].mean()) if p70.sum() else np.nan,
                    mean_plddt=float(plddt[ok].mean()) if ok.sum() else np.nan,
                    status="ok")
                log(f"  {tag}: n={row['n_res']} (p70 {row['n_res_p70']}) "
                    f"f_buried={row['f_buried']:.3f} / p70 {row['f_buried_p70']:.3f}")
            except Exception as exc:
                row["status"] = f"error: {exc.__class__.__name__}: {exc}"
                log(f"  {tag}: FAILED {row['status']}")
                with open(LOG, "a") as fh:
                    fh.write(traceback.format_exc() + "\n")

        pd.DataFrame([row]).to_csv(CSV, mode="a", header=header_needed, index=False)
        header_needed = False

    d = pd.read_csv(CSV)
    ok = d[d.status == "ok"]
    log(f"DONE: {len(ok)}/{len(d)} families succeeded; "
        f"f_buried_p70 median {ok.f_buried_p70.median():.3f} "
        f"(range {ok.f_buried_p70.min():.3f}-{ok.f_buried_p70.max():.3f})")
    bad = d[d.status != "ok"]
    if len(bad):
        log(f"failures ({len(bad)}): " + ", ".join(f"{r.unit} [{r.status[:40]}]"
                                                   for _, r in bad.iterrows()))


if __name__ == "__main__":
    main()
