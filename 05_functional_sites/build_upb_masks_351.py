#!/usr/bin/env python3
"""Step B1 (fast, no ChimeraX): for each UPb-covered gene-named family, pick the reference AF3
structure, extract its sequence+resids, fetch reviewed-cyano UniProt sites, align, and map sites
onto reference residue numbering. Writes upb_sites_351.csv (unit, ref_path, ResID, is_site) +
upb_sites_summary_351.csv. Run in mrcr env (Bio.Align, gemmi)."""
import os, sys, glob, json, subprocess, time
import pandas as pd
from Bio import Align

ROOT = "/home/yangyicheng/27.evolution"
OUT = "/home/yangyicheng/27.evolution/40.whole_CR-MR"
sys.path.insert(0, os.path.join(ROOT, "28.mrcr-20250602"))
import structure_mr_cr_ChimeraX as S  # read_cif_structure

SITE_TYPES = {"Active site", "Binding site", "Metal binding", "Site"}
aligner = Align.PairwiseAligner()
aligner.mode = "global"; aligner.open_gap_score = -10; aligner.extend_gap_score = -0.5
aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")

pan = pd.read_csv(f"{OUT}/panorama_families.csv")
cov = pd.read_csv("/tmp/claude-3055/-home-yangyicheng-27-evolution/a63e0cc0-e9cf-4e01-8ca8-5158fcc17151/scratchpad/upb_coverage.csv")
g95 = set(cov[cov.n_site_feat > 0].gene)
sub = pan[pan.unit.isin(g95)].copy()


def base_dir(tree, module):
    return os.path.join(ROOT, "27.allophycocyanin") if tree == "27.allophycocyanin" else os.path.join(ROOT, tree, str(module))


def cif_nres(p):
    try:
        return sum(1 for ln in open(p) if ln.startswith("ATOM") and " CA " in ln)
    except Exception:
        return 0


def pick_ref(unit_dir):
    for gcf in ("GCF_000009725", "GCF_000011345"):
        for pat in (f"WP_*_{gcf}/*/seed-1_sample-0/model.cif", f"WP_*_{gcf}/*/*_model.cif"):
            g = glob.glob(os.path.join(unit_dir, pat))
            if g:
                return g[0]
    cands = glob.glob(os.path.join(unit_dir, "WP_*_GCF_*/*/seed-1_sample-0/model.cif")) or \
        glob.glob(os.path.join(unit_dir, "WP_*_GCF_*/*/*_model.cif"))
    return max(cands, key=cif_nres) if cands else None


def fetch(name):
    url = ("https://rest.uniprot.org/uniprotkb/search?query=gene:" + name +
           "%20AND%20taxonomy_id:1117%20AND%20reviewed:true"
           "&fields=accession,organism_name,sequence,ft_act_site,ft_binding,ft_site&format=json&size=25")
    try:
        return json.loads(subprocess.run(["curl", "-s", "--max-time", "30", url],
                                          capture_output=True, text=True).stdout).get("results", [])
    except Exception:
        return []


rows, summ = [], []
for _, r in sub.iterrows():
    unit, tree, module = r["unit"], r["tree"], r["module"]
    unit_dir = os.path.join(base_dir(tree, module), unit)
    ref = pick_ref(unit_dir)
    if not ref:
        summ.append((unit, tree, "-", "no_ref_struct", 0, 0)); continue
    try:
        ref_seq, _, ref_ids, _ = S.read_cif_structure(ref)
    except Exception as e:
        summ.append((unit, tree, ref, f"read_fail:{e}", 0, 0)); continue
    resids = [rid[0] for rid in ref_ids]
    cands = fetch(unit)
    best = None
    for c in cands:
        useq = c.get("sequence", {}).get("value", "")
        if len(useq) < 30:
            continue
        aln = aligner.align(ref_seq, useq)[0]
        nid = sum(a == b for a, b in zip(str(aln[0]), str(aln[1])) if a != "-" and b != "-")
        ident = nid / min(len(ref_seq), len(useq))
        if best is None or ident > best[0]:
            best = (ident, c, aln)
    if best is None:
        summ.append((unit, tree, ref, "no_uniprot_seq", 0, len(resids))); continue
    ident, c, aln = best
    u2r, ri, ui = {}, 0, 0
    for a, b in zip(str(aln[0]), str(aln[1])):
        if a != "-" and b != "-":
            u2r[ui] = ri
        if a != "-":
            ri += 1
        if b != "-":
            ui += 1
    sites = set()
    for f in c.get("features", []):
        if f.get("type") not in SITE_TYPES:
            continue
        s, e = f["location"]["start"]["value"], f["location"]["end"]["value"]
        for pos in range(s, e + 1):
            if (pos - 1) in u2r:
                sites.add(resids[u2r[pos - 1]])
    for rid in resids:
        rows.append((unit, ref, rid, int(rid in sites)))
    summ.append((unit, tree, c["primaryAccession"], f"id={ident:.2f}", len(sites), len(resids)))
    print(f"  {unit:8} {tree:20} acc={c['primaryAccession']:8} id={ident:.2f} sites={len(sites):3}/{len(resids):3}")
    time.sleep(0.2)

pd.DataFrame(rows, columns=["unit", "ref_path", "ResID", "is_site"]).to_csv(f"{OUT}/upb_sites_351.csv", index=False)
sdf = pd.DataFrame(summ, columns=["unit", "tree", "acc", "status", "n_sites", "n_res"])
sdf.to_csv(f"{OUT}/upb_sites_summary_351.csv", index=False)
ok = sdf[sdf.n_sites > 0]
print(f"\n=== UPb masks built: {len(ok)}/{len(sub)} families got >=1 site mapped to reference ===")
print("failures:", sdf[sdf.n_sites == 0].status.value_counts().to_dict())
