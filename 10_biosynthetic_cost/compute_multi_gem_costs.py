#!/usr/bin/env python3
"""Multi-GEM amino-acid cost calculation (extends single-iJN678 pipeline).

Closes caveat #4 (single-GEM proxy): computes the same 20 AA × 5 baseline-
subtracted cost metrics across 3 published BiGG-convention cyanobacteria GEMs
(2 species, 3 reconstructions), then produces a consensus table + cross-model
agreement statistics.

Design notes
------------
* Standalone — does NOT modify compute_cyano_costs.py or its outputs/. All new
  results go under outputs_multigem/. Old single-GEM results are untouched.
* Config-list driven (MODEL_CONFIGS). Adding the 3 non-BiGG GEMs later
  (PCC7002 / iTeryR / iSP1101) only needs a new config entry with metabolite-ID
  overrides (aa_met / atp_met / nadph_met / *_rxns) — no algorithm change.
* iJB785 splits photon into 15 wavelength exchanges; total photon cost = their
  sum and the photon-fixing step uses a pooled flux constraint.

Run
---
    PY=/home/yangyicheng/mambaforge/envs/orthofinder/bin/python   # or py12
    $PY compute_multi_gem_costs.py
"""
from __future__ import annotations

import os
import sys
import warnings
import logging
from pathlib import Path

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba

from cyano_cost_lib import AA_MET as DEFAULT_AA_MET, AA_MW, AA_ORDER

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "outputs_multigem"          # NEW dir — never touches outputs/
BY_MODEL_DIR = OUT_DIR / "by_model"
OUT_DIR.mkdir(parents=True, exist_ok=True)
BY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

GEM_DIR = Path("/home/yangyicheng/27.evolution/17.flux_CR/08.models")

# Shared BiGG ion-uptake set (same metabolite IDs across the 3 BiGG models)
DEFAULT_ION_UPTAKE = [
    "EX_pi_e", "EX_so4_e", "EX_h2o_e", "EX_h_e", "EX_o2_e", "EX_ca2_e",
    "EX_cobalt2_e", "EX_ni2_e", "EX_k_e", "EX_mobd_e", "EX_fe3_e", "EX_fe2_e",
    "EX_na1_e", "EX_mn2_e", "EX_mg2_e", "EX_cu2_e", "EX_zn2_e",
]

# ---------------------------------------------------------------------------
# Model registry. 3 BiGG models now; non-BiGG entries can be appended later.
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        "name": "iJN678",
        "species": "Synechocystis sp. PCC 6803",
        "gcf": "GCF_000009725",
        "ref": "Nogales 2012",
        "model_file": str(GEM_DIR / "iJN678.xml"),
        "format": "sbml",
        "biomass_id": "BIOMASS_Ec_SynAuto",
        "photon_rxns": ["EX_photon_e"],
        "photon_pattern": None,
        "co2_rxns": ["EX_co2_e", "EX_hco3_e"],
        "n_rxns": {"NH4": "EX_nh4_e", "NO3": "EX_no3_e"},
        "atp_met": "atp_c",
        "nadph_met": "nadph_c",
        "aa_met": None,                 # None -> DEFAULT_AA_MET (BiGG)
        "ion_uptake": None,             # None -> DEFAULT_ION_UPTAKE
    },
    {
        "name": "iSynCJ816",
        "species": "Synechocystis sp. PCC 6803",
        "gcf": "GCF_000009725",
        "ref": "Joshi 2017",
        "model_file": str(GEM_DIR / "iSynCJ816.json"),
        "format": "json",
        "biomass_id": "BIOMASS_Ec_SynAuto_1",
        "photon_rxns": ["EX_photon_e"],
        "photon_pattern": None,
        "co2_rxns": ["EX_co2_e", "EX_hco3_e"],
        "n_rxns": {"NH4": "EX_nh4_e", "NO3": "EX_no3_e"},
        "atp_met": "atp_c",
        "nadph_met": "nadph_c",
        "aa_met": None,
        "ion_uptake": None,
    },
    {
        "name": "iJB785",
        "species": "Synechococcus elongatus PCC 7942",
        "gcf": "GCF_000012525",
        "ref": "Broddrick 2016",
        "model_file": str(GEM_DIR / "iJB785.xml"),
        "format": "sbml",
        "biomass_id": "BIOMASS__1",
        "photon_rxns": None,            # resolved from pattern at load
        "photon_pattern": ("EX_photon", "_e"),   # startswith, endswith
        "co2_rxns": ["EX_co2_e", "EX_hco3_e"],
        "n_rxns": {"NH4": "EX_nh4_e", "NO3": "EX_no3_e"},
        "atp_met": "atp_c",
        "nadph_met": "nadph_c",
        "aa_met": None,
        "ion_uptake": None,
    },
]

CORE_METRICS = [
    "Growth-coupled photon cost",
    "Inorganic carbon cost",
    "ATP turnover demand",
    "NADPH turnover demand",
    "N assimilation demand",
]


# ---------------------------------------------------------------------------
# Model IO + config resolution
# ---------------------------------------------------------------------------
def _load_model(cfg: dict) -> cobra.Model:
    with open(os.devnull, "w") as dn:
        old = sys.stderr
        sys.stderr = dn
        try:
            if cfg["format"] == "json":
                m = cobra.io.load_json_model(cfg["model_file"])
            else:
                m = cobra.io.read_sbml_model(cfg["model_file"])
        finally:
            sys.stderr = old
    return m


def _resolve_photon_rxns(model: cobra.Model, cfg: dict) -> list[str]:
    if cfg.get("photon_rxns"):
        return [r for r in cfg["photon_rxns"] if r in model.reactions]
    pat = cfg.get("photon_pattern")
    if pat:
        prefix, suffix = pat
        return [r.id for r in model.reactions
                if r.id.startswith(prefix) and r.id.endswith(suffix)]
    return []


def _aa_met_map(cfg: dict) -> dict:
    return cfg["aa_met"] if cfg.get("aa_met") else DEFAULT_AA_MET


def _ion_uptake(cfg: dict) -> list[str]:
    return cfg["ion_uptake"] if cfg.get("ion_uptake") else DEFAULT_ION_UPTAKE


# ---------------------------------------------------------------------------
# FBA helpers (parametrized versions of compute_cyano_costs.py helpers)
# ---------------------------------------------------------------------------
def _configure_photoautotrophic(model: cobra.Model, n_source: str, cfg: dict) -> None:
    photon_rxns = _resolve_photon_rxns(model, cfg)
    for ex in model.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0
    # open photon + carbon + ions
    for rid in photon_rxns + cfg["co2_rxns"] + _ion_uptake(cfg):
        if rid in model.reactions:
            model.reactions.get_by_id(rid).lower_bound = -1000.0
    # N source
    nh4 = cfg["n_rxns"].get("NH4")
    no3 = cfg["n_rxns"].get("NO3")
    if nh4 and nh4 in model.reactions:
        model.reactions.get_by_id(nh4).lower_bound = -1000.0 if n_source == "NH4" else 0.0
    if no3 and no3 in model.reactions:
        model.reactions.get_by_id(no3).lower_bound = -1000.0 if n_source == "NO3" else 0.0
    # force photoautotrophy: block glucose + AA import
    for rid in ("EX_glc__D_e", "EX_glc_e"):
        if rid in model.reactions:
            model.reactions.get_by_id(rid).lower_bound = 0.0
            model.reactions.get_by_id(rid).upper_bound = 0.0
    for ex in model.exchanges:
        if ex.id.startswith("EX_") and "__L_e" in ex.id:
            ex.lower_bound = 0.0
        if ex.id == "EX_gly_e":
            ex.lower_bound = 0.0


def _add_dm(model: cobra.Model, met_id: str, rid: str) -> cobra.Reaction:
    dm = cobra.Reaction(rid)
    dm.lower_bound = 1.0
    dm.upper_bound = 1.0
    dm.add_metabolites({model.metabolites.get_by_id(met_id): -1.0})
    model.add_reactions([dm])
    return dm


def _minimize_uptake(model: cobra.Model, uptake_ids: list[str]) -> tuple[float, object]:
    valid = [rid for rid in uptake_ids if rid in model.reactions]
    if not valid:
        return np.nan, None
    expr = 0
    for rid in valid:
        rxn = model.reactions.get_by_id(rid)
        rxn.upper_bound = min(0.0, rxn.upper_bound)
        expr = expr + rxn.flux_expression
    model.objective = expr
    sol = model.optimize(objective_sense="max")
    if sol.status != "optimal":
        return np.nan, sol
    return -float(sum(sol.fluxes[rid] for rid in valid)), sol


def _fix_total_uptake(model: cobra.Model, uptake_ids: list[str], total_uptake: float) -> None:
    """Constrain net uptake (sum of -flux) to `total_uptake`.
    Single reaction -> set bounds directly. Multiple -> add pooled constraint."""
    valid = [rid for rid in uptake_ids if rid in model.reactions]
    if not valid:
        return
    target_flux = -float(total_uptake)   # uptake is negative flux
    if len(valid) == 1:
        rxn = model.reactions.get_by_id(valid[0])
        rxn.lower_bound = target_flux - 1e-6
        rxn.upper_bound = target_flux + 1e-6
        return
    expr = sum(model.reactions.get_by_id(rid).flux_expression for rid in valid)
    cons = model.problem.Constraint(expr, lb=target_flux - 1e-6, ub=target_flux + 1e-6,
                                    name="pooled_uptake_fix")
    model.add_cons_vars(cons)


def _met_turnover(model: cobra.Model, sol, met_id: str) -> float:
    used = 0.0
    for rxn in model.reactions:
        v = float(sol.fluxes.get(rxn.id, 0.0))
        if abs(v) <= 1e-12:
            continue
        for met, coeff in rxn.metabolites.items():
            if met.id != met_id:
                continue
            used += max(0.0, -coeff * v)
    return float(used)


def _norm_col(x: pd.Series) -> pd.Series:
    vals = pd.to_numeric(x, errors="coerce")
    ok = vals.notna()
    out = pd.Series(np.nan, index=x.index, dtype=float)
    if ok.sum() <= 1:
        return out
    mn, mx = float(vals[ok].min()), float(vals[ok].max())
    out[ok] = 0.0 if (mx - mn <= 1e-12) else (vals[ok] - mn) / (mx - mn)
    return out


def _find_biomass_rxn(model: cobra.Model, biomass_id: str | None) -> str:
    rxn_ids = {r.id for r in model.reactions}
    if biomass_id:
        if biomass_id in rxn_ids:
            return biomass_id
        raise RuntimeError(f"biomass id not found: {biomass_id}")
    try:
        coeffs = model.objective.expression.as_coefficients_dict()
        contrib: dict[str, float] = {}
        for var, c in coeffs.items():
            name = getattr(var, "name", None)
            if not name:
                continue
            base = name.rsplit("_reverse_", 1)[0]
            if base in rxn_ids:
                contrib[base] = contrib.get(base, 0.0) + abs(float(c))
        if contrib:
            return max(contrib.items(), key=lambda kv: kv[1])[0]
    except Exception:
        pass
    cands = [r.id for r in model.reactions if "biomass" in r.id.lower()]
    if len(cands) == 1:
        return cands[0]
    raise RuntimeError(f"cannot resolve biomass; candidates={cands}")


# ---------------------------------------------------------------------------
# Per-model cost table (mirrors compute_cyano_costs.compute_cost_table)
# ---------------------------------------------------------------------------
def compute_cost_table(cfg: dict, growth_lb: float = 0.05, n_source: str = "NH4") -> pd.DataFrame:
    base = _load_model(cfg)
    bio_id = _find_biomass_rxn(base, cfg.get("biomass_id"))
    aa_met = _aa_met_map(cfg)
    photon_rxns = _resolve_photon_rxns(base, cfg)
    atp_met, nadph_met = cfg["atp_met"], cfg["nadph_met"]
    n_uptake_rxns = [r for r in (cfg["n_rxns"].get("NH4"), cfg["n_rxns"].get("NO3")) if r]

    # --- growth-only baselines ---
    m0 = base.copy()
    _configure_photoautotrophic(m0, n_source, cfg)
    b0 = m0.reactions.get_by_id(bio_id)
    b0.lower_bound = b0.upper_bound = growth_lb
    baseline_photon, _ = _minimize_uptake(m0, photon_rxns)

    baseline_atp = baseline_nadph = baseline_n = np.nan
    if np.isfinite(baseline_photon) and photon_rxns:
        _fix_total_uptake(m0, photon_rxns, baseline_photon)
        m0.objective = m0.reactions.get_by_id(bio_id)
        s0 = pfba(m0)
        baseline_atp = _met_turnover(m0, s0, atp_met)
        baseline_nadph = _met_turnover(m0, s0, nadph_met)
        baseline_n = sum(max(0.0, -float(s0.fluxes[r])) for r in n_uptake_rxns if r in m0.reactions)

    m0c = base.copy()
    _configure_photoautotrophic(m0c, n_source, cfg)
    b0c = m0c.reactions.get_by_id(bio_id)
    b0c.lower_bound = b0c.upper_bound = growth_lb
    baseline_carbon, _ = _minimize_uptake(m0c, cfg["co2_rxns"])

    rows = []
    base_mets = {m.id for m in base.metabolites}
    for aa in AA_ORDER:
        met = aa_met[aa]
        row = {"aa": aa, "metabolite": met, "growth_lb": growth_lb,
               "N_source": n_source, "model": cfg["name"]}
        if met not in base_mets:
            rows.append(row)
            continue

        # photon cost
        m2 = base.copy()
        _configure_photoautotrophic(m2, n_source, cfg)
        b2 = m2.reactions.get_by_id(bio_id)
        b2.lower_bound = b2.upper_bound = growth_lb
        dm2 = _add_dm(m2, met, f"DM_{met}_gphoton")
        photon_aa, sol2 = _minimize_uptake(m2, photon_rxns)
        row["Growth-coupled photon cost"] = (
            np.nan if not (np.isfinite(photon_aa) and np.isfinite(baseline_photon))
            else max(0.0, photon_aa - baseline_photon)
        )

        # ATP / NADPH / N from photon-minimal state
        if sol2 is not None and getattr(sol2, "status", "") == "optimal" and photon_rxns:
            _fix_total_uptake(m2, photon_rxns, photon_aa)
            m2.objective = dm2
            s2 = pfba(m2)
            atp_raw = _met_turnover(m2, s2, atp_met)
            nadph_raw = _met_turnover(m2, s2, nadph_met)
            n_raw = sum(max(0.0, -float(s2.fluxes[r])) for r in n_uptake_rxns if r in m2.reactions)
            row["ATP turnover demand"] = np.nan if not np.isfinite(baseline_atp) else max(0.0, atp_raw - baseline_atp)
            row["NADPH turnover demand"] = np.nan if not np.isfinite(baseline_nadph) else max(0.0, nadph_raw - baseline_nadph)
            row["N assimilation demand"] = np.nan if not np.isfinite(baseline_n) else max(0.0, n_raw - baseline_n)
        else:
            row["ATP turnover demand"] = np.nan
            row["NADPH turnover demand"] = np.nan
            row["N assimilation demand"] = np.nan

        # inorganic carbon cost
        m3 = base.copy()
        _configure_photoautotrophic(m3, n_source, cfg)
        b3 = m3.reactions.get_by_id(bio_id)
        b3.lower_bound = b3.upper_bound = growth_lb
        _add_dm(m3, met, f"DM_{met}_carb")
        carb_aa, _ = _minimize_uptake(m3, cfg["co2_rxns"])
        row["Inorganic carbon cost"] = (
            np.nan if not (np.isfinite(carb_aa) and np.isfinite(baseline_carbon))
            else max(0.0, carb_aa - baseline_carbon)
        )
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("aa").reset_index(drop=True)
    df["Weight"] = df["aa"].map(AA_MW).astype(float)
    for c in CORE_METRICS:
        df[f"{c} (norm)"] = _norm_col(df[c])
    df["Cyano averaged cost"] = df[[f"{c} (norm)" for c in CORE_METRICS]].mean(axis=1, skipna=False)
    return df


# ---------------------------------------------------------------------------
# Consensus + agreement
# ---------------------------------------------------------------------------
def build_consensus(per_model: dict[str, pd.DataFrame], n_source: str) -> pd.DataFrame:
    """Median (+mean+CV) of raw metrics across models, then re-normalize."""
    names = list(per_model)
    idx = per_model[names[0]].set_index("aa")
    out = pd.DataFrame({"aa": idx.index})
    out["metabolite"] = idx["metabolite"].values
    out["N_source"] = n_source
    out["n_models"] = len(names)
    out["Weight"] = out["aa"].map(AA_MW).astype(float)

    for c in CORE_METRICS:
        stack = pd.concat([per_model[n].set_index("aa")[c] for n in names], axis=1)
        out[c] = stack.median(axis=1).values
        out[f"{c} (mean)"] = stack.mean(axis=1).values
        sd = stack.std(axis=1, ddof=0)
        mn = stack.mean(axis=1)
        out[f"{c} (CV)"] = (sd / mn.replace(0, np.nan)).values

    for c in CORE_METRICS:
        out[f"{c} (norm)"] = _norm_col(out[c]).values
    out["Cyano averaged cost"] = out[[f"{c} (norm)" for c in CORE_METRICS]].mean(axis=1, skipna=False)
    return out


def build_agreement(per_model: dict[str, pd.DataFrame], n_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import spearmanr
    names = list(per_model)
    cy = {n: per_model[n].set_index("aa")["Cyano averaged cost"] for n in names}

    pair_rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            d = pd.DataFrame({"a": cy[a], "b": cy[b]}).dropna()
            if len(d) >= 3:
                rho, p = spearmanr(d["a"], d["b"])
            else:
                rho, p = np.nan, np.nan
            pair_rows.append({"N_source": n_source, "model_A": a, "model_B": b,
                              "n_AA": len(d), "spearman_rho": rho, "p": p})
    pairs = pd.DataFrame(pair_rows)

    # per-AA CV across models on Cyano avg cost
    stack = pd.concat([cy[n] for n in names], axis=1)
    aa_rows = []
    for aa in stack.index:
        vals = stack.loc[aa].dropna().values
        if len(vals) >= 2:
            mean = float(np.mean(vals))
            cv = float(np.std(vals, ddof=0) / mean) if mean else np.nan
        else:
            mean, cv = np.nan, np.nan
        aa_rows.append({"N_source": n_source, "aa": aa, "n_models": len(vals),
                        "mean_cyano_avg": mean, "cv_across_models": cv})
    per_aa = pd.DataFrame(aa_rows).sort_values("cv_across_models", ascending=False)
    return pairs, per_aa


# ---------------------------------------------------------------------------
def main() -> None:
    all_pairs, all_aa = [], []
    for n_source in ("NH4", "NO3"):
        print(f"\n=== N source: {n_source} ===")
        per_model: dict[str, pd.DataFrame] = {}
        for cfg in MODEL_CONFIGS:
            print(f"[run] {cfg['name']} ({cfg['ref']}, {cfg['species']}) ...")
            df = compute_cost_table(cfg, growth_lb=0.05, n_source=n_source)
            per_model[cfg["name"]] = df
            fp = BY_MODEL_DIR / f"cyano_aa_costs_{cfg['name']}_{n_source}.tsv"
            df.to_csv(fp, sep="\t", index=False)
            print(f"      -> {fp.name}")
            top = df.sort_values("Cyano averaged cost").tail(1)["aa"].iloc[0]
            bot = df.sort_values("Cyano averaged cost").head(1)["aa"].iloc[0]
            print(f"      cheapest={bot}  priciest={top}")

        cons = build_consensus(per_model, n_source)
        cons_fp = OUT_DIR / f"cyano_aa_costs_consensus_{n_source}.tsv"
        cons.to_csv(cons_fp, sep="\t", index=False)
        print(f"[OK] consensus -> {cons_fp.name}")

        pairs, per_aa = build_agreement(per_model, n_source)
        all_pairs.append(pairs)
        all_aa.append(per_aa)

    pd.concat(all_pairs, ignore_index=True).to_csv(
        OUT_DIR / "cyano_aa_cost_model_agreement_pairs.tsv", sep="\t", index=False)
    pd.concat(all_aa, ignore_index=True).to_csv(
        OUT_DIR / "cyano_aa_cost_model_agreement_per_aa.tsv", sep="\t", index=False)
    print(f"\n[OK] agreement tables -> {OUT_DIR}")
    print("[DONE] old outputs/ untouched; new results under outputs_multigem/")


if __name__ == "__main__":
    main()
