#!/usr/bin/env python3
"""Step B2 (heavy, ChimeraX): for each UPb-covered family, compute MR/CR for four masks
{complete, core, surface, binding} against every member, using the canonical structure_mr_cr
pipeline (2.0 A Ca mapping, RSA<0.25 = core). Reference + core/surface reproduced here so all
four categories are self-consistent. Checkpointed (skips existing outputs). Run in mrcr env."""
import os, sys, glob, json, csv, tempfile, shutil
import numpy as np, pandas as pd

ROOT = "/home/yangyicheng/27.evolution"
OUT = "/home/yangyicheng/27.evolution/40.whole_CR-MR"
RESDIR = os.path.join(OUT, "Result_binding_upb")
os.makedirs(RESDIR, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "28.mrcr-20250602"))
import structure_mr_cr_ChimeraX as S

RSA_CUT = 0.25
pan = pd.read_csv(f"{OUT}/panorama_families.csv")
unit2tm = {r.unit: (r.tree, r.module) for r in pan.itertuples()}
sites = pd.read_csv(f"{OUT}/upb_sites_351.csv")
site_ids = {u: set(d[d.is_site == 1].ResID) for u, d in sites.groupby("unit")}
ref_of = dict(zip(sites.unit, sites.ref_path))
units = [u for u in site_ids if len(site_ids[u]) > 0]


def base_dir(tree, module):
    return os.path.join(ROOT, "27.allophycocyanin") if tree == "27.allophycocyanin" else os.path.join(ROOT, tree, str(module))


def targets_of(unit):
    tree, module = unit2tm[unit]
    ud = os.path.join(base_dir(tree, module), unit)
    g = glob.glob(os.path.join(ud, "WP_*_GCF_*/*/seed-1_sample-0/model.cif")) or \
        glob.glob(os.path.join(ud, "WP_*_GCF_*/*/*_model.cif"))
    return g


def ref_masks(ref_path, unit, temp_dir):
    """Return ref_seq, ref_ca, and masks dict aligned to reference residue order."""
    sp, jp = S.compute_properties_with_chimerax(ref_path, temp_dir=temp_dir)
    if not sp:
        return None
    ref_seq, ref_ca, ref_ids, _ = S.read_cif_structure(sp)
    props = {(it['seqid'], it['icode'].strip()): (it['sasa'], it['ss_type']) for it in json.load(open(jp))}
    sasa = np.array([props.get((rid[0], rid[1].strip()), (0.0, 0))[0] for rid in ref_ids])
    rsa = S.get_rsa(ref_seq, sasa)
    sset = site_ids[unit]
    masks = {
        'complete': np.ones(len(ref_seq), dtype=bool),
        'core': rsa < RSA_CUT,
        'surface': rsa >= RSA_CUT,
        'binding': np.array([rid[0] in sset for rid in ref_ids]),
    }
    return ref_seq, ref_ca, masks


MASK_ORDER = ['complete', 'core', 'surface', 'binding']


def main():
    print(f"[INFO] {len(units)} UPb families to compute")
    for i, unit in enumerate(sorted(units)):
        out = os.path.join(RESDIR, f"{unit}_mrcr_results.csv")
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            print(f"[SKIP] {unit}"); continue
        ref = ref_of[unit]
        tgts = [t for t in targets_of(unit) if os.path.realpath(t) != os.path.realpath(ref)]
        if not os.path.isfile(ref) or len(tgts) < 2:
            print(f"[FAIL] {unit}: ref={os.path.isfile(ref)} targets={len(tgts)}"); continue
        temp_dir = tempfile.mkdtemp(dir=RESDIR, prefix=f".tmp_{unit}_")
        try:
            rm = ref_masks(ref, unit, temp_dir)
            if rm is None:
                print(f"[FAIL] {unit}: ref SASA failed"); continue
            ref_seq, ref_ca, masks = rm
            nb = int(masks['binding'].sum())
            header = ["ref", "target", "target_file"] + [f"{k}_{m}" for k in MASK_ORDER for m in ("MR", "CR")]
            rows = [header]
            for tf in tgts:
                try:
                    ap = os.path.join(temp_dir, f"aln_{os.path.basename(os.path.dirname(os.path.dirname(tf)))}.cif")
                    fa = S.call_chimerax_matchmaker(ref, tf, ap, cutoff=2.0)
                    if not fa:
                        raise RuntimeError("align fail")
                    tseq, tca, _, _ = S.read_cif_structure(fa)
                    res = S.compute_mr_cr_for_target(ref_seq, ref_ca, tseq, tca, masks, cutoff=2.0)
                    row = [S.extract_protein_name(ref), S.extract_protein_name(tf), os.path.basename(tf)]
                    for k in MASK_ORDER:
                        row += [f"{res.get(f'{k}_MR',0.0):.4f}", f"{res.get(f'{k}_CR',0.0):.4f}"]
                    rows.append(row)
                except Exception as e:
                    rows.append([S.extract_protein_name(ref), S.extract_protein_name(tf), os.path.basename(tf)] + ["NA"] * 8)
            with open(out, "w", newline="") as f:
                csv.writer(f).writerows(rows)
            print(f"[OK  ] {unit} ({i+1}/{len(units)}) targets={len(tgts)} binding_res={nb} -> {out}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    print("[DONE]")


if __name__ == "__main__":
    main()
