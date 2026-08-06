#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path("/home/yangyicheng/27.evolution/38.protein_cost")
WORK = ROOT / "Cyane_cost_cal"
OUT_DIR = WORK / "outputs"
PANEL_DIR = OUT_DIR / "panels"
DATA_ROOT = Path("/home/yangyicheng/27.evolution/00.data")
PAIRWISE = ROOT / "protein_cost_pairwise.csv"
ABUNDANCE = Path("/home/yangyicheng/27.evolution/19.pepMutic_CR/abundance_merged.csv")
MODEL_FILE = ROOT / "iJN678.xml"
CORE_SURFACE_DIR = Path("/home/yangyicheng/27.evolution/15.coresiteAminoAcid/02.plot2/101.CRcore_surfacee")

OUT_DIR.mkdir(parents=True, exist_ok=True)
PANEL_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDED_ASSEMBLIES = {
    "GCF_000011545.1",  # Burkholderia (non-cyano, mistakenly in data dir)
}


AA_MET = {
    "A": "ala__L_c",
    "R": "arg__L_c",
    "N": "asn__L_c",
    "D": "asp__L_c",
    "C": "cys__L_c",
    "Q": "gln__L_c",
    "E": "glu__L_c",
    "G": "gly_c",
    "H": "his__L_c",
    "I": "ile__L_c",
    "L": "leu__L_c",
    "K": "lys__L_c",
    "M": "met__L_c",
    "F": "phe__L_c",
    "P": "pro__L_c",
    "S": "ser__L_c",
    "T": "thr__L_c",
    "W": "trp__L_c",
    "Y": "tyr__L_c",
    "V": "val__L_c",
}
AA_ORDER = sorted(AA_MET.keys())
AA_MW = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16, "Q": 146.15, "E": 147.13,
    "G": 75.07, "H": 155.16, "I": 131.17, "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19,
    "P": 115.13, "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}


def load_seq_index() -> Dict[str, str]:
    out: Dict[str, str] = {}
    fasta_files = list(DATA_ROOT.glob("*/ncbi_dataset/data/*/protein.faa"))
    # Filter out non-cyano assemblies (e.g. Burkholderia accidentally in data dir)
    fasta_files = [fp for fp in fasta_files
                   if not any(ex in str(fp) for ex in EXCLUDED_ASSEMBLIES)]
    for fp in fasta_files:
        header = None
        seq: List[str] = []
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                if ln.startswith(">"):
                    if header is not None:
                        m = re.search(r"(WP_\d+)", header)
                        if m and m.group(1) not in out:
                            out[m.group(1)] = "".join(seq).upper()
                    header = ln[1:]
                    seq = []
                else:
                    seq.append(ln)
            if header is not None:
                m = re.search(r"(WP_\d+)", header)
                if m and m.group(1) not in out:
                    out[m.group(1)] = "".join(seq).upper()
    return out


def load_core_surface_cr() -> pd.DataFrame:
    files = list(CORE_SURFACE_DIR.glob("*.csv"))
    rows = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        target_col = "target_wp" if "target_wp" in d.columns else ("target" if "target" in d.columns else None)
        if target_col is None:
            continue
        if not {"core_CR", "surface_CR"}.issubset(d.columns):
            continue
        x = d[[target_col, "core_CR", "surface_CR"]].copy()
        x = x.rename(columns={target_col: "target_wp_raw"})
        x["target_wp_norm"] = x["target_wp_raw"].astype(str).str.extract(r"(WP_\d+)")[0]
        x["core_CR"] = pd.to_numeric(x["core_CR"], errors="coerce")
        x["surface_CR"] = pd.to_numeric(x["surface_CR"], errors="coerce")
        x = x.dropna(subset=["target_wp_norm"])
        x = x[np.isfinite(x["core_CR"]) & np.isfinite(x["surface_CR"])]
        rows.append(x[["target_wp_norm", "core_CR", "surface_CR"]])
    if not rows:
        return pd.DataFrame(columns=["target_wp_norm", "core_CR", "surface_CR"])
    out = pd.concat(rows, ignore_index=True)
    out = out.groupby("target_wp_norm", as_index=False)[["core_CR", "surface_CR"]].mean()
    return out


def merge_core_surface(df: pd.DataFrame) -> pd.DataFrame:
    cs = load_core_surface_cr()
    out = df.copy()
    out["target_wp_norm"] = out["target_wp"].astype(str).str.extract(r"(WP_\d+)")[0]
    if cs.empty:
        out["core_CR"] = np.nan
        out["surface_CR"] = np.nan
    else:
        out = out.merge(cs, on="target_wp_norm", how="left")
    return out


def add_abundance(df: pd.DataFrame) -> pd.DataFrame:
    if not ABUNDANCE.exists():
        df["log2_abundance"] = np.nan
        return df
    ab = pd.read_csv(ABUNDANCE)
    id_col = "Protein_ID" if "Protein_ID" in ab.columns else ("protein_id" if "protein_id" in ab.columns else None)
    
    val_col = None
    if "Mean_log2_abundance" in ab.columns:
        val_col = "Mean_log2_abundance"
    elif "mean_log2_abundance" in ab.columns:
        val_col = "mean_log2_abundance"
    elif "Abundance" in ab.columns:
        val_col = "Abundance"
        
    if id_col is None or val_col is None:
        df["log2_abundance"] = np.nan
        return df
        
    ab = ab[[id_col, val_col]].dropna()
    
    if val_col == "Abundance":
        ab["log2_abundance_val"] = np.log2(ab[val_col].replace(0, np.nan))
    else:
        ab["log2_abundance_val"] = ab[val_col]
        
    ab["wp"] = ab[id_col].astype(str).str.extract(r"(WP_\d+)")[0]
    ab = ab.dropna(subset=["wp", "log2_abundance_val"])
    ab = ab.groupby("wp", as_index=False)["log2_abundance_val"].mean()
    ref_map = ab.rename(columns={"wp": "ref_wp", "log2_abundance_val": "log2_abundance_ref"})
    tgt_map = ab.rename(columns={"wp": "target_wp", "log2_abundance_val": "log2_abundance_tgt"})
    x = df.merge(ref_map, on="ref_wp", how="left")
    x = x.merge(tgt_map, on="target_wp", how="left")
    x["log2_abundance"] = x["log2_abundance_ref"]
    m = x["log2_abundance"].isna()
    x.loc[m, "log2_abundance"] = x.loc[m, "log2_abundance_tgt"]
    return x.drop(columns=["log2_abundance_ref", "log2_abundance_tgt"])


def add_aa_fraction(df: pd.DataFrame, seq_idx: Dict[str, str]) -> pd.DataFrame:
    aas = list("ACDEFGHIKLMNPQRSTVWY")
    fr = {f"frac_{aa}": [] for aa in aas}
    for wp in df["target_wp"]:
        seq = seq_idx.get(str(wp), "")
        L = len(seq)
        if L == 0:
            for aa in aas:
                fr[f"frac_{aa}"].append(np.nan)
            continue
        for aa in aas:
            fr[f"frac_{aa}"].append(seq.count(aa) / L)
    for k, v in fr.items():
        df[k] = v
    return df
