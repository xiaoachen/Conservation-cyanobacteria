#!/usr/bin/env python3
from __future__ import annotations

import cobra
import numpy as np
import pandas as pd
from cobra.flux_analysis import pfba

from cyano_cost_lib import AA_MET, AA_MW, AA_ORDER, MODEL_FILE, OUT_DIR

DEFAULT_BIOMASS_ID = "BIOMASS_Ec_SynAuto"  # cyano photoautotrophic biomass


def _configure_photoautotrophic(model: cobra.Model, n_source: str) -> None:
    for ex in model.exchanges:
        ex.lower_bound = 0.0
        ex.upper_bound = 1000.0
    open_uptake = {
        "EX_photon_e": 1000.0,
        "EX_hco3_e": 1000.0,
        "EX_co2_e": 1000.0,
        "EX_pi_e": 1000.0,
        "EX_so4_e": 1000.0,
        "EX_h2o_e": 1000.0,
        "EX_h_e": 1000.0,
        "EX_o2_e": 1000.0,
        "EX_ca2_e": 1000.0,
        "EX_cobalt2_e": 1000.0,
        "EX_ni2_e": 1000.0,
        "EX_k_e": 1000.0,
        "EX_mobd_e": 1000.0,
        "EX_fe3_e": 1000.0,
        "EX_fe2_e": 1000.0,
        "EX_na1_e": 1000.0,
        "EX_mn2_e": 1000.0,
        "EX_mg2_e": 1000.0,
        "EX_cu2_e": 1000.0,
        "EX_zn2_e": 1000.0,
    }
    for rid, v in open_uptake.items():
        if rid in model.reactions:
            model.reactions.get_by_id(rid).lower_bound = -v
    # explicit N-source condition
    if "EX_nh4_e" in model.reactions:
        model.reactions.get_by_id("EX_nh4_e").lower_bound = -1000.0 if n_source == "NH4" else 0.0
    if "EX_no3_e" in model.reactions:
        model.reactions.get_by_id("EX_no3_e").lower_bound = -1000.0 if n_source == "NO3" else 0.0
    # force photoautotrophy
    if "EX_glc__D_e" in model.reactions:
        model.reactions.get_by_id("EX_glc__D_e").lower_bound = 0.0
        model.reactions.get_by_id("EX_glc__D_e").upper_bound = 0.0
    # block AA import
    for ex in model.exchanges:
        rid = ex.id
        if rid.startswith("EX_") and "__L_e" in rid:
            ex.lower_bound = 0.0
        if rid == "EX_gly_e":
            ex.lower_bound = 0.0


def _add_dm(model: cobra.Model, met_id: str, rid: str) -> cobra.Reaction:
    dm = cobra.Reaction(rid)
    dm.lower_bound = 1.0
    dm.upper_bound = 1.0
    dm.add_metabolites({model.metabolites.get_by_id(met_id): -1.0})
    model.add_reactions([dm])
    return dm


def _find_biomass_rxn(model: cobra.Model, biomass_id: str | None = None) -> str:
    rxn_ids = {r.id for r in model.reactions}
    # Priority 1: explicit override
    if biomass_id:
        if biomass_id in rxn_ids:
            return biomass_id
        raise RuntimeError(f"Specified biomass id not found in model: {biomass_id}")
    # Priority 2: model.objective single/max-coeff reaction
    try:
        coeffs = model.objective.expression.as_coefficients_dict()
        contrib: dict[str, float] = {}
        for var, c in coeffs.items():
            name = getattr(var, "name", None)
            if not name:
                continue
            base = name[:-len("_reverse_5d8af")] if "_reverse_" in name else name
            base = base.rsplit("_reverse_", 1)[0]
            if base in rxn_ids:
                contrib[base] = contrib.get(base, 0.0) + abs(float(c))
        if len(contrib) == 1:
            return next(iter(contrib))
        if len(contrib) > 1:
            return max(contrib.items(), key=lambda kv: kv[1])[0]
    except Exception:
        pass
    # Priority 3: unique "biomass" id/name match
    cands = [
        r.id for r in model.reactions
        if "biomass" in r.id.lower() or "biomass" in (r.name or "").lower()
    ]
    if len(cands) == 1:
        return cands[0]
    raise RuntimeError(
        f"Cannot uniquely resolve biomass reaction; candidates={cands}. "
        f"Specify biomass_id explicitly."
    )


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
    mn = float(vals[ok].min())
    mx = float(vals[ok].max())
    if mx - mn <= 1e-12:
        out[ok] = 0.0
    else:
        out[ok] = (vals[ok] - mn) / (mx - mn)
    return out


def compute_cost_table(
    growth_lb: float = 0.05,
    n_source: str = "NH4",
    biomass_id: str | None = DEFAULT_BIOMASS_ID,
) -> pd.DataFrame:
    base = cobra.io.read_sbml_model(str(MODEL_FILE))
    bio_id = _find_biomass_rxn(base, biomass_id=biomass_id)

    # growth-only baseline photon
    m0 = base.copy()
    _configure_photoautotrophic(m0, n_source=n_source)
    bio0 = m0.reactions.get_by_id(bio_id)
    bio0.lower_bound = growth_lb
    bio0.upper_bound = growth_lb
    baseline_photon, _ = _minimize_uptake(m0, ["EX_photon_e"])

    baseline_atp = np.nan
    baseline_nadph = np.nan
    baseline_n_uptake = np.nan
    if np.isfinite(baseline_photon) and "EX_photon_e" in m0.reactions:
        ex_ph0 = m0.reactions.get_by_id("EX_photon_e")
        target_flux0 = -float(baseline_photon)
        ex_ph0.lower_bound = target_flux0 - 1e-6
        ex_ph0.upper_bound = target_flux0 + 1e-6
        m0.objective = m0.reactions.get_by_id(bio_id)
        sol0_pf = pfba(m0)
        baseline_atp = _met_turnover(m0, sol0_pf, "atp_c")
        baseline_nadph = _met_turnover(m0, sol0_pf, "nadph_c")
        n_uptake0 = 0.0
        for rid in ("EX_no3_e", "EX_nh4_e"):
            if rid in m0.reactions:
                n_uptake0 += max(0.0, -float(sol0_pf.fluxes[rid]))
        baseline_n_uptake = n_uptake0

    # growth-only baseline inorganic carbon uptake (same growth_lb, no AA demand)
    baseline_carbon = np.nan
    m0c = base.copy()
    _configure_photoautotrophic(m0c, n_source=n_source)
    bio0c = m0c.reactions.get_by_id(bio_id)
    bio0c.lower_bound = growth_lb
    bio0c.upper_bound = growth_lb
    baseline_carbon, _ = _minimize_uptake(m0c, ["EX_co2_e", "EX_hco3_e"])

    rows = []
    for aa in AA_ORDER:
        met = AA_MET[aa]
        row = {"aa": aa, "metabolite": met, "growth_lb": growth_lb, "N_source": n_source}
        if met not in [m.id for m in base.metabolites]:
            rows.append(row)
            continue

        # growth-coupled photon cost minus growth-only baseline
        m2 = base.copy()
        _configure_photoautotrophic(m2, n_source=n_source)
        bio2 = m2.reactions.get_by_id(bio_id)
        bio2.lower_bound = growth_lb
        bio2.upper_bound = growth_lb
        dm2 = _add_dm(m2, met, f"DM_{met}_gphoton")
        photon_with_growth_aa, sol2 = _minimize_uptake(m2, ["EX_photon_e"])
        row["Growth+AA photon"] = photon_with_growth_aa
        row["Growth-only photon baseline"] = baseline_photon
        row["Growth-coupled photon cost"] = (
            np.nan if not np.isfinite(photon_with_growth_aa) else max(0.0, photon_with_growth_aa - baseline_photon)
        )

        # 3/4/5/6 from photon-minimal state: fix photon flux then pFBA
        if sol2 is not None and getattr(sol2, "status", "") == "optimal" and "EX_photon_e" in m2.reactions:
            ex_ph = m2.reactions.get_by_id("EX_photon_e")
            target_flux = -float(photon_with_growth_aa)
            ex_ph.lower_bound = target_flux - 1e-6
            ex_ph.upper_bound = target_flux + 1e-6
            m2.objective = dm2
            sol2_pf = pfba(m2)
            atp_raw = _met_turnover(m2, sol2_pf, "atp_c")
            nadph_raw = _met_turnover(m2, sol2_pf, "nadph_c")
            n_uptake = 0.0
            for rid in ("EX_no3_e", "EX_nh4_e"):
                if rid in m2.reactions:
                    n_uptake += max(0.0, -float(sol2_pf.fluxes[rid]))
            row["ATP turnover (growth+AA)"] = atp_raw
            row["ATP turnover (growth-only baseline)"] = baseline_atp
            row["ATP turnover demand"] = np.nan if not np.isfinite(baseline_atp) else max(0.0, atp_raw - baseline_atp)
            row["NADPH turnover (growth+AA)"] = nadph_raw
            row["NADPH turnover (growth-only baseline)"] = baseline_nadph
            row["NADPH turnover demand"] = (
                np.nan if not np.isfinite(baseline_nadph) else max(0.0, nadph_raw - baseline_nadph)
            )
            row["Total N uptake (growth+AA)"] = n_uptake
            row["Total N uptake (growth-only baseline)"] = baseline_n_uptake
            row["N assimilation demand"] = (
                np.nan if not np.isfinite(baseline_n_uptake) else max(0.0, n_uptake - baseline_n_uptake)
            )
        else:
            row["ATP turnover demand"] = np.nan
            row["NADPH turnover demand"] = np.nan
            row["N assimilation demand"] = np.nan

        # 7) growth-coupled inorganic carbon cost (baseline-subtracted)
        m3 = base.copy()
        _configure_photoautotrophic(m3, n_source=n_source)
        bio3 = m3.reactions.get_by_id(bio_id)
        bio3.lower_bound = growth_lb
        bio3.upper_bound = growth_lb
        _add_dm(m3, met, f"DM_{met}_carb")
        carb_with_growth_aa, _ = _minimize_uptake(m3, ["EX_co2_e", "EX_hco3_e"])
        row["Carbon uptake (growth+AA)"] = carb_with_growth_aa
        row["Carbon uptake (growth-only baseline)"] = baseline_carbon
        row["Inorganic carbon cost"] = (
            np.nan
            if not (np.isfinite(carb_with_growth_aa) and np.isfinite(baseline_carbon))
            else max(0.0, carb_with_growth_aa - baseline_carbon)
        )

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("aa")
    df["Weight"] = df["aa"].map(AA_MW).astype(float)
    core = [
        "Growth-coupled photon cost",
        "Inorganic carbon cost",
        "ATP turnover demand",
        "NADPH turnover demand",
        "N assimilation demand",
    ]
    for c in core:
        df[f"{c} (norm)"] = _norm_col(df[c])
    df["Cyano averaged cost"] = df[[f"{c} (norm)" for c in core]].mean(axis=1, skipna=False)
    return df


def main() -> None:
    # Run both N source conditions; keep cyano_aa_costs.tsv = NH4 for downstream compat.
    print(f"[INFO] biomass reaction: {DEFAULT_BIOMASS_ID}")
    aa_nh4 = compute_cost_table(
        growth_lb=0.05, n_source="NH4", biomass_id=DEFAULT_BIOMASS_ID
    )
    aa_nh4.to_csv(OUT_DIR / "cyano_aa_costs_NH4.tsv", sep="\t", index=False)
    aa_nh4.to_csv(OUT_DIR / "cyano_aa_costs.tsv", sep="\t", index=False)
    print("[OK] wrote:", OUT_DIR / "cyano_aa_costs_NH4.tsv")

    aa_no3 = compute_cost_table(
        growth_lb=0.05, n_source="NO3", biomass_id=DEFAULT_BIOMASS_ID
    )
    aa_no3.to_csv(OUT_DIR / "cyano_aa_costs_NO3.tsv", sep="\t", index=False)
    print("[OK] wrote:", OUT_DIR / "cyano_aa_costs_NO3.tsv")


if __name__ == "__main__":
    main()
