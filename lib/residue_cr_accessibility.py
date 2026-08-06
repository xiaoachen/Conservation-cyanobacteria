#!/usr/bin/env python3
"""Residue-level quantification: mutational ACCESSIBILITY (FoldX) vs CONSERVATION.

Per reference residue:
  accessibility  = FoldX whole-protein scan: mean ddG over 19 substitutions, and
                   FSV = fraction of substitutions with ddG < theta (structurally tolerated variants)
  conservation   = identity-to-reference across orthologs (MSA column), the per-residue analog of CR
                   (project CR = of mapped residues, fraction identical)

Pools residues across the analysed proteins -> N = hundreds (vs n=6 at protein level), so a
proper Spearman with power. Reference numbering = FoldX {gene}_Repair.pdb (1-based).

Usage: residue_cr_accessibility.py <gene1> [gene2 ...]
"""
import os, sys, glob, subprocess, tempfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

MAX_SASA = {'A':121.,'R':265.,'N':187.,'D':187.,'C':148.,'E':214.,'Q':214.,'G':97.,'H':216.,
            'I':195.,'L':191.,'K':230.,'M':214.,'F':228.,'P':154.,'S':143.,'T':163.,'W':264.,
            'Y':255.,'V':165.}


def compute_rsa(pdb):
    """Per-residue relative solvent accessibility from the reference structure (Bio.PDB Shrake-Rupley)."""
    from Bio.PDB import PDBParser
    from Bio.PDB.SASA import ShrakeRupley
    s = PDBParser(QUIET=True).get_structure("x", pdb)
    ShrakeRupley().compute(s, level="R")
    out = {}
    for res in s[0].get_residues():
        nm = res.resname.strip()
        one = THREE2ONE.get(nm)
        if one and res.id[0] == " ":
            out[res.id[1]] = res.sasa / MAX_SASA.get(one, 200.0)
    return out


def partial_spearman(x, y, z):
    """Partial Spearman of x,y controlling z (rank residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    def resid(a, b):
        B = np.c_[np.ones_like(b), b]
        c, *_ = np.linalg.lstsq(B, a, rcond=None)
        return a - B @ c
    return pearsonr(resid(rx, rz), resid(ry, rz))

ROOT = "/home/yangyicheng/27.evolution"
FOLDX = f"{ROOT}/37.AGT-RMSD-entropy/05.whole_prot_foldx"   # has all 10 genes (Repair.pdb + scan)
MRCR = f"{ROOT}/12.mrcr-20251125"                           # ortholog AF3 models per module/gene
OUT = f"{ROOT}/20.AGT-RMSD-entropy/06.residue_accessibility_CR"
MAFFT = "/home/yangyicheng/mambaforge/envs/orthofinder/bin/mafft"
THETA = 2.0   # ddG threshold (kcal/mol) for "tolerated" -> FSV

# two enzyme types (dual-mechanism, RESULTS_SUMMARY §0.3):
#  I  = catalytic-core / single-residue-constrained (small, catalytically dense)
#  II = supercomplex-interface / large multidomain (CR high but not single-residue driven)
TYPE = {"psbA": "I", "psbC": "I", "psbD": "I", "petC": "I", "petE": "I", "petJ": "I",
        "glpX": "I", "tpiA": "I", "rpiA": "I", "rpe": "I",                 # Type I (n=10)
        "psaA": "II", "atpA": "II", "atpD": "II", "psaB": "II", "psaC": "II",
        "psaD": "II", "atpB": "II", "petA": "II", "petB": "II", "petD": "II"}   # Type II (n=10)
ONE = {v: k for k, v in {'A':'ALA','R':'ARG','N':'ASN','D':'ASP','C':'CYS','Q':'GLN','E':'GLU',
       'G':'GLY','H':'HIS','I':'ILE','L':'LEU','K':'LYS','M':'MET','F':'PHE','P':'PRO','S':'SER',
       'T':'THR','W':'TRP','Y':'TYR','V':'VAL'}.items()}

THREE2ONE = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H',
    'ILE':'I','LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V'}


def ref_seq_from_pdb(pdb):
    """Reference one-letter sequence + ordered residue numbers from CA records."""
    seq, resids = [], []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            rn = THREE2ONE.get(line[17:20].strip(), "X")
            seq.append(rn); resids.append(int(line[22:26]))
    return "".join(seq), resids


def seq_from_cif(cif):
    """Extract first protein chain one-letter sequence from a model.cif (CA order)."""
    import gemmi
    st = gemmi.read_structure(cif)
    for ch in st[0]:
        s = []
        for res in ch:
            if res.get_ca() is None:
                continue
            try:
                o = gemmi.find_tabulated_residue(res.name).one_letter_code.upper()
            except Exception:
                o = "X"
            if o.isalpha():
                s.append(o)
        if len(s) > 30:
            return "".join(s)
    return None


def find_ortholog_cifs(gene):
    # one representative AF3 model per ortholog (wp_*_model.cif), searching all 12.mrcr modules
    for base in glob.glob(f"{MRCR}/*/{gene}"):
        if os.path.isdir(base):
            cifs = glob.glob(f"{base}/**/*_model.cif", recursive=True)
            if cifs:
                return cifs
    return []


def per_residue_conservation(gene, ref_seq, ref_resids):
    """MSA orthologs to reference; per ref residue = fraction of orthologs identical to ref."""
    cifs = find_ortholog_cifs(gene)
    seqs = {}
    for c in cifs:
        # unique key per ortholog = the WP_*_GCF_* folder (2 levels above the cif)
        key = os.path.basename(os.path.dirname(os.path.dirname(c)))
        if key in seqs:
            continue
        s = seq_from_cif(c)
        if s:
            seqs[key] = s
    print(f"  [{gene}] orthologs with sequence: {len(seqs)}")
    fa = os.path.join(OUT, f"_{gene}.fa")
    with open(fa, "w") as f:
        f.write(f">REF\n{ref_seq}\n")
        for k, s in seqs.items():
            f.write(f">{k}\n{s}\n")
    aln = os.path.join(OUT, f"_{gene}.aln")
    with open(aln, "w") as fo:
        subprocess.run([MAFFT, "--auto", "--quiet", fa], stdout=fo, check=True)
    # read alignment
    names, alns = [], []
    cur = None
    for line in open(aln):
        line = line.rstrip()
        if line.startswith(">"):
            names.append(line[1:]); alns.append([]); cur = alns[-1]
        elif cur is not None:
            cur.append(line)
    seqs_aln = ["".join(a) for a in alns]
    ref_aln = seqs_aln[names.index("REF")]
    others = [seqs_aln[i] for i, n in enumerate(names) if n != "REF"]
    # map alignment columns -> reference residue index
    cons = {}
    ri = 0
    for col, rc in enumerate(ref_aln):
        if rc == "-":
            continue
        col_chars = [o[col] for o in others if o[col] != "-"]
        ident = sum(1 for c in col_chars if c == rc)
        n_obs = len(col_chars)
        cons[ref_resids[ri]] = (ident / n_obs if n_obs else np.nan, n_obs)
        ri += 1
    return cons


import re
_SCAN = re.compile(r"^([A-Z]{3})([A-Za-z])(\d+)([A-Z])\t([-\d.eE]+)")


def per_residue_accessibility(gene):
    """Parse FoldX position-scan PS_<gene>_Repair_scanning_output.txt -> per-residue mean ddG + FSV.
    Line format: <WT3><chain><resid><mut1>\\t<ddG>; skip the WT->WT self entry."""
    path = f"{FOLDX}/{gene}/PS_{gene}_Repair_scanning_output.txt"
    rows = []
    for line in open(path, errors="ignore"):
        m = _SCAN.match(line.strip())
        if not m:
            continue
        wt3, _ch, resid, mut, ddg = m.groups()
        if ONE.get(wt3) == mut:          # skip self-substitution (ddG=0)
            continue
        try:
            rows.append((int(resid), float(ddg)))
        except ValueError:
            continue
    d = pd.DataFrame(rows, columns=["ResID", "ddG"])
    g = d.groupby("ResID")["ddG"].agg(mean_ddG="mean", n_mut="count").reset_index()
    g["FSV"] = d.groupby("ResID")["ddG"].apply(lambda s: (s < THETA).mean()).values
    return g


def analyze(gene):
    ref_seq, ref_resids = ref_seq_from_pdb(f"{FOLDX}/{gene}/{gene}_Repair.pdb")
    cons = per_residue_conservation(gene, ref_seq, ref_resids)
    acc = per_residue_accessibility(gene)
    acc["CR_res"] = acc["ResID"].map(lambda r: cons.get(r, (np.nan,))[0])
    acc["n_orth"] = acc["ResID"].map(lambda r: cons.get(r, (np.nan, np.nan))[1])
    rsa = compute_rsa(f"{FOLDX}/{gene}/{gene}_Repair.pdb")
    acc["RSA"] = acc["ResID"].map(rsa)
    acc["Gene"] = gene
    acc["Type"] = TYPE.get(gene, "?")
    return acc.dropna(subset=["CR_res", "mean_ddG"])


def main():
    genes = sys.argv[1:] or list(TYPE.keys())   # all 19 (10 Type I + 9 Type II)
    allres = []
    for g in genes:
        try:
            r = analyze(g)
            rho_d, p_d = spearmanr(r["mean_ddG"], r["CR_res"])
            rho_f, p_f = spearmanr(r["FSV"], r["CR_res"])
            print(f"[{g}] N={len(r)}  mean_ddG vs CR: ρ={rho_d:+.3f} p={p_d:.2e} | "
                  f"FSV vs CR: ρ={rho_f:+.3f} p={p_f:.2e}")
            allres.append(r)
        except Exception as e:
            print(f"[{g}] FAILED: {e}")
    if allres:
        df = pd.concat(allres, ignore_index=True)
        df.to_csv(f"{OUT}/residue_accessibility_CR.csv", index=False)
        rho_d, p_d = spearmanr(df["mean_ddG"], df["CR_res"])
        rho_f, p_f = spearmanr(df["FSV"], df["CR_res"])
        print(f"\n=== POOLED N={len(df)} ({df['Gene'].nunique()} proteins) ===")
        print(f"  mean ΔΔG vs CR : ρ={rho_d:+.3f}  p={p_d:.2e}")
        print(f"  FSV       vs CR : ρ={rho_f:+.3f}  p={p_f:.2e}")
        # burial (RSA) confound control
        sub = df.dropna(subset=["RSA"])
        rho_rsa, _ = spearmanr(sub["RSA"], sub["CR_res"])
        pr_d = partial_spearman(sub["mean_ddG"], sub["CR_res"], sub["RSA"])
        pr_f = partial_spearman(sub["FSV"], sub["CR_res"], sub["RSA"])
        print(f"\n  RSA(burial) vs CR : ρ={rho_rsa:+.3f}  (buried→conserved confound)")
        print(f"  partial ΔΔG vs CR | RSA : ρ={pr_d[0]:+.3f}  p={pr_d[1]:.2e}")
        print(f"  partial FSV vs CR | RSA : ρ={pr_f[0]:+.3f}  p={pr_f[1]:.2e}")
        # two enzyme types contrast
        print("\n=== by enzyme type (I=single-residue/catalytic, II=interface/large) ===")
        for t in ["I", "II"]:
            s = df[df.Type == t]
            rd, pd_ = spearmanr(s.mean_ddG, s.CR_res)
            rf, pf_ = spearmanr(s.FSV, s.CR_res)
            print(f"  Type {t}: N={len(s)} ({s.Gene.nunique()} prot)  ΔΔG×CR ρ={rd:+.3f} p={pd_:.1e} | "
                  f"FSV×CR ρ={rf:+.3f} p={pf_:.1e}")
        print(f"  saved -> {OUT}/residue_accessibility_CR.csv")


if __name__ == "__main__":
    main()
