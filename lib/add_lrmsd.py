#!/usr/bin/env python3
"""Per-residue lRMSD (structural drift across orthologs) for all 35 genes — 37's structural dimension,
brought to residue level. For each reference residue: RMS of its Cα deviation across ortholog AF3
models, after Kabsch superposition using the MSA-derived residue correspondence.

Reuses residue_cr_accessibility (ref seq/resids, ortholog cif lookup, FOLDX paths, MSA .aln).
Run in mrcr env (gemmi). Output: residue_lrmsd.csv (Gene, ResID, lRMSD, n_models).
"""
import os, glob
import numpy as np
import residue_cr_accessibility as R

GENES = list(R.TYPE.keys()) + ["OG0000957", "OG0001060", "OG0000971", "OG0001028", "OG0001064",
                               "OG0001089", "OG0001081", "OG0000960", "OG0001054", "OG0001037",
                               "OG0001048", "OG0000993", "OG0001044", "OG0001024", "OG0001077"]


def ca_coords_from_pdb(pdb):
    """ref CA coords in CA order + resids (mirror ref_seq_from_pdb ordering)."""
    xyz, resids = [], []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            xyz.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            resids.append(int(line[22:26]))
    return np.array(xyz), resids


def ca_coords_from_cif(cif):
    """ortholog CA coords in the SAME order/filter as residue_cr_accessibility.seq_from_cif."""
    import gemmi
    st = gemmi.read_structure(cif)
    for ch in st[0]:
        coords = []
        for res in ch:
            ca = res.get_ca()
            if ca is None:
                continue
            try:
                o = gemmi.find_tabulated_residue(res.name).one_letter_code.upper()
            except Exception:
                o = "X"
            if o.isalpha():
                coords.append([ca.pos.x, ca.pos.y, ca.pos.z])
        if len(coords) > 30:
            return np.array(coords)
    return None


def kabsch(P, Q):
    """rotate+translate P onto Q (both Nx3); return transformed P."""
    Pc, Qc = P.mean(0), Q.mean(0)
    H = (P - Pc).T @ (Q - Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rm = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (P - Pc) @ Rm.T + Qc


def read_aln(path):
    names, seqs, cur = [], [], None
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            names.append(line[1:]); seqs.append([]); cur = seqs[-1]
        elif cur is not None:
            cur.append(line)
    return names, ["".join(s) for s in seqs]


def per_residue_lrmsd(gene):
    refpdb = f"{R.FOLDX}/{gene}/{gene}_Repair.pdb"
    aln = f"{R.OUT}/_{gene}.aln"
    if not (os.path.exists(refpdb) and os.path.exists(aln)):
        return None
    ref_xyz, ref_resids = ca_coords_from_pdb(refpdb)
    # ortholog name -> cif coords
    name2cif = {os.path.basename(os.path.dirname(os.path.dirname(c))): c
                for c in R.find_ortholog_cifs(gene)}
    names, alns = read_aln(aln)
    if "REF" not in names:
        return None
    ref_aln = alns[names.index("REF")]
    dev = {r: [] for r in ref_resids}     # ref resid -> list of Cα deviations across orthologs
    nmod = 0
    for nm, aseq in zip(names, alns):
        if nm == "REF" or nm not in name2cif:
            continue
        oc = ca_coords_from_cif(name2cif[nm])
        if oc is None:
            continue
        # walk columns: matched (ref_ca_idx, orth_ca_idx) where both non-gap
        ri = oi = -1
        pairs = []
        for cr, co in zip(ref_aln, aseq):
            if cr != "-":
                ri += 1
            if co != "-":
                oi += 1
            if cr != "-" and co != "-" and ri < len(ref_xyz) and oi < len(oc):
                pairs.append((ri, oi))
        if len(pairs) < 20:
            continue
        ridx = [p[0] for p in pairs]; oidx = [p[1] for p in pairs]
        P, Q = oc[oidx], ref_xyz[ridx]
        Pt = kabsch(P, Q)
        d = np.linalg.norm(Pt - Q, axis=1)
        for k, ri_ in enumerate(ridx):
            dev[ref_resids[ri_]].append(d[k])
        nmod += 1
    rows = [(gene, r, float(np.sqrt(np.mean(np.square(v)))), len(v)) for r, v in dev.items() if v]
    print(f"  [{gene}] lRMSD over {nmod} ortholog models, {len(rows)} residues")
    return rows


def main():
    import pandas as pd
    allrows = []
    for g in GENES:
        try:
            r = per_residue_lrmsd(g)
            if r:
                allrows += r
        except Exception as e:
            print(f"  [{g}] FAIL {e}")
    df = pd.DataFrame(allrows, columns=["Gene", "ResID", "lRMSD", "n_models"])
    df.to_csv(f"{R.OUT}/residue_lrmsd.csv", index=False)
    print(f"saved residue_lrmsd.csv: {len(df)} residues, {df.Gene.nunique()} genes; "
          f"lRMSD median={df.lRMSD.median():.2f} Å")


if __name__ == "__main__":
    main()
