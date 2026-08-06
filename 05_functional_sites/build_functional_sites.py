#!/usr/bin/env python3
"""Independent functional-site annotations per enzyme from UniProt (CR-independent), for all 35 genes.
For each gene: query reviewed cyanobacterial (taxon 1117) entries, pick the candidate whose sequence
best matches our FoldX reference, align it to the reference, and map UniProt functional sites
(Active site / Binding site / Metal binding / Site) onto our reference residue numbering.

Output: functional_sites.csv (Gene, ResID, is_site)  +  prints coverage per gene.
Run in mrcr env (Bio.Align). Needs network (UniProt REST).
"""
import json, subprocess, time
import pandas as pd
from Bio import Align
import residue_cr_accessibility as R

GENES = list(R.TYPE.keys()) + list(__import__("analyze_core").CORE.keys())
SITE_TYPES = {"Active site", "Binding site", "Metal binding", "Site"}
aligner = Align.PairwiseAligner()
aligner.mode = "global"
aligner.open_gap_score = -10
aligner.extend_gap_score = -0.5
aligner.substitution_matrix = Align.substitution_matrices.load("BLOSUM62")


def ref_seq_resids(gene):
    seq, resids = [], []
    for ln in open(f"{R.FOLDX}/{gene}/{gene}_Repair.pdb"):
        if ln.startswith("ATOM") and ln[12:16].strip() == "CA":
            seq.append(R.THREE2ONE.get(ln[17:20].strip(), "X")); resids.append(int(ln[22:26]))
    return "".join(seq), resids


def gene_label(gene):
    if gene.startswith("OG"):
        return __import__("analyze_core").CORE[gene][1]
    return gene


def fetch_candidates(name):
    url = ("https://rest.uniprot.org/uniprotkb/search?query=gene:" + name +
           "%20AND%20taxonomy_id:1117%20AND%20reviewed:true"
           "&fields=accession,organism_name,sequence,ft_act_site,ft_binding,ft_site&format=json&size=25")
    try:
        out = subprocess.run(["curl", "-s", "--max-time", "30", url], capture_output=True, text=True).stdout
        return json.loads(out).get("results", [])
    except Exception as e:
        print(f"    fetch fail {name}: {e}"); return []


def map_sites(gene):
    ref, resids = ref_seq_resids(gene)
    name = gene_label(gene)
    cands = fetch_candidates(name)
    if not cands:
        return [], None, 0.0, name, "no UniProt hit"
    # pick candidate with best identity to ref
    best = None
    for c in cands:
        useq = c.get("sequence", {}).get("value", "")
        if len(useq) < 30:
            continue
        aln = aligner.align(ref, useq)[0]
        nid = sum(a == b for a, b in zip(str(aln[0]), str(aln[1])) if a != "-" and b != "-")
        ident = nid / min(len(ref), len(useq))
        if best is None or ident > best[0]:
            best = (ident, c, useq, aln)
    if best is None:
        return [], None, 0.0, name, "no usable seq"
    ident, c, useq, aln = best
    # uniprot 1-based pos -> ref index, via alignment coordinate blocks
    u2r = {}
    ri = ui = 0
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
        s = f["location"]["start"]["value"]; e = f["location"]["end"]["value"]
        for pos in range(s, e + 1):              # uniprot 1-based
            if (pos - 1) in u2r:
                sites.add(resids[u2r[pos - 1]])
    return sorted(sites), c["primaryAccession"], ident, name, c.get("organism", {}).get("scientificName", "")


def main():
    rows, summ = [], []
    for g in GENES:
        sites, acc, ident, name, org = map_sites(g)
        # mark every ref residue
        refseq, resids = ref_seq_resids(g)
        sset = set(sites)
        for r in resids:
            rows.append((g, r, int(r in sset)))
        summ.append((g, name, acc, f"{ident:.2f}" if acc else "-", len(sites), len(resids), org[:32]))
        print(f"  {name:6}({g:10}) acc={acc or '-':10} id={ident:.2f} sites={len(sites):3}/{len(resids):3}  {org[:30]}")
        time.sleep(0.3)
    df = pd.DataFrame(rows, columns=["Gene", "ResID", "is_site"])
    df.to_csv(f"{R.OUT}/functional_sites.csv", index=False)
    s = pd.DataFrame(summ, columns=["gene", "name", "acc", "ident", "n_sites", "n_res", "organism"])
    s.to_csv(f"{R.OUT}/functional_sites_summary.csv", index=False)
    nz = s[s.n_sites > 0]
    print(f"\nsaved functional_sites.csv: {df.is_site.sum()} site-residues; "
          f"{len(nz)}/{len(s)} genes have >=1 annotated site (median id={s.ident.replace('-',None).astype(float).median():.2f})")


if __name__ == "__main__":
    main()
