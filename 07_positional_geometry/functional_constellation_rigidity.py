#!/usr/bin/env python3
"""Functional-constellation geometric rigidity (generalisable P) — all genes with reliable UniProt sites.

For each gene: take its UniProt functional-site residues (the 'functional constellation'); in every
ortholog AF3 model measure all pairwise Cα distances among those sites; the coefficient of variation
(CV = std/mean) of each pairwise distance across orthologs quantifies how geometrically LOCKED that
pair is. GLI (geometric-lockdown index) = mean CV over pairs (LOW = locked). Pairwise distances are
superposition-invariant, so no Kabsch needed. Pure coordinate variance -> non-circular w.r.t. CR.

Reuses residue_cr_accessibility (R) + add_lrmsd machinery (MSA mapping, ortholog cif lookup).
Run in mrcr env (gemmi). Output: functional_constellation_rigidity.csv.
"""
import os
import numpy as np, pandas as pd
import residue_cr_accessibility as R
from add_lrmsd import ca_coords_from_pdb, ca_coords_from_cif, read_aln

MIN_MODELS = 20          # per pair, need this many ortholog models with both sites present
OUT = R.OUT


def constellation_cv(gene, site_resids):
    refpdb = f"{R.FOLDX}/{gene}/{gene}_Repair.pdb"
    aln = f"{OUT}/_{gene}.aln"
    if not (os.path.exists(refpdb) and os.path.exists(aln)):
        return None
    ref_xyz, ref_resids = ca_coords_from_pdb(refpdb)
    resid2idx = {r: i for i, r in enumerate(ref_resids)}
    sites = [s for s in site_resids if s in resid2idx]          # sites present in the reference CA set
    if len(sites) < 2:
        return None
    site_refidx = [resid2idx[s] for s in sites]
    name2cif = {os.path.basename(os.path.dirname(os.path.dirname(c))): c
                for c in R.find_ortholog_cifs(gene)}
    names, alns = read_aln(aln)
    if "REF" not in names:
        return None
    ref_aln = alns[names.index("REF")]
    # per ortholog model: coord of each functional site (by ref CA index) if mapped & present
    per_model_coords = []                                       # list of dict{ref_ca_idx: xyz}
    for nm, aseq in zip(names, alns):
        if nm == "REF" or nm not in name2cif:
            continue
        oc = ca_coords_from_cif(name2cif[nm])
        if oc is None:
            continue
        ri = oi = -1
        ref2orth = {}
        for cr, co in zip(ref_aln, aseq):
            if cr != "-":
                ri += 1
            if co != "-":
                oi += 1
            if cr != "-" and co != "-" and ri < len(ref_xyz) and oi < len(oc):
                ref2orth[ri] = oi
        coords = {ridx: oc[ref2orth[ridx]] for ridx in site_refidx if ridx in ref2orth}
        if len(coords) >= 2:
            per_model_coords.append(coords)
    if len(per_model_coords) < MIN_MODELS:
        return None
    # per pair of sites: distances across models -> CV
    cvs = []
    for a in range(len(site_refidx)):
        for b in range(a + 1, len(site_refidx)):
            ia, ib = site_refidx[a], site_refidx[b]
            dists = [np.linalg.norm(m[ia] - m[ib]) for m in per_model_coords if ia in m and ib in m]
            if len(dists) >= MIN_MODELS:
                dists = np.array(dists)
                cvs.append(dists.std() / dists.mean())
    if not cvs:
        return None
    nmod = len(per_model_coords)
    return dict(Gene=gene, n_sites=len(sites), n_pairs=len(cvs),
                n_models=nmod, GLI_meanCV=float(np.mean(cvs)), GLI_medianCV=float(np.median(cvs)))


def main():
    fs = pd.read_csv(f"{OUT}/functional_sites.csv")
    summ = pd.read_csv(f"{OUT}/functional_sites_summary.csv")
    summ["ident"] = pd.to_numeric(summ["ident"], errors="coerce")
    good = summ[(summ.ident >= 0.7) & (summ.n_sites >= 1)].gene.tolist()
    rows = []
    for g in good:
        sr = fs[(fs.Gene == g) & (fs.is_site == 1)].ResID.tolist()
        try:
            r = constellation_cv(g, sr)
        except Exception as e:
            print(f"  [{g}] FAIL {e}"); r = None
        if r:
            rows.append(r); print(f"  [{g}] sites={r['n_sites']} pairs={r['n_pairs']} "
                                   f"models={r['n_models']} GLI(meanCV)={r['GLI_meanCV']:.4f}")
        else:
            print(f"  [{g}] skipped (insufficient mapping/models)")
    df = pd.DataFrame(rows)
    # attach protein-level CR proxy = mean per-residue CR (r=0.94 with official CR)
    A = pd.read_csv(f"{OUT}/residue_accessibility_CR.csv")
    crmean = A.groupby("Gene").CR_res.mean().rename("CR_protein")
    ddg = A.groupby("Gene").mean_ddG.mean().rename("ddG_protein")
    lr = pd.read_csv(f"{OUT}/residue_lrmsd.csv").groupby("Gene").lRMSD.mean().rename("lRMSD_protein")
    df = df.merge(crmean, left_on="Gene", right_index=True, how="left") \
           .merge(ddg, left_on="Gene", right_index=True, how="left") \
           .merge(lr, left_on="Gene", right_index=True, how="left")
    df.to_csv(f"{OUT}/functional_constellation_rigidity.csv", index=False)
    print(f"\nsaved functional_constellation_rigidity.csv: {len(df)} genes")
    # correlations
    from scipy.stats import spearmanr, rankdata, pearsonr
    d = df.dropna(subset=["GLI_meanCV", "CR_protein"])
    print(f"\nn={len(d)} genes")
    for c in ["GLI_meanCV", "GLI_medianCV"]:
        r, p = spearmanr(d[c], d.CR_protein)
        print(f"  {c} vs CR_protein : rho={r:+.3f}  p={p:.3f}  (locked geometry, expect NEGATIVE)")
    r, p = spearmanr(d.lRMSD_protein, d.CR_protein); print(f"  lRMSD_protein vs CR : rho={r:+.3f} p={p:.3f}")
    r, p = spearmanr(d.ddG_protein, d.CR_protein);   print(f"  ddG_protein   vs CR : rho={r:+.3f} p={p:.3f}")
    # partial GLI vs CR | ddG (is geometric lockdown independent of energy?)
    def partial(x, y, z):
        rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
        res = lambda a, b: a - np.c_[np.ones_like(b), b] @ np.linalg.lstsq(np.c_[np.ones_like(b), b], a, rcond=None)[0]
        return pearsonr(res(rx, rz), res(ry, rz))
    dd = d.dropna(subset=["ddG_protein"])
    r, p = partial(dd.GLI_meanCV, dd.CR_protein, dd.ddG_protein)
    print(f"  GLI vs CR | ddG      : rho={r:+.3f}  p={p:.3f}  (geometric lockdown independent of energy?)")


if __name__ == "__main__":
    main()
