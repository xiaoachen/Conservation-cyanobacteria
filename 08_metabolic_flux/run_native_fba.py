#!/usr/bin/env python3
import numpy as np
"""Run light + dark pFBA on each downloadable cyano GEM and emit per-strain
flux TSV with gene→OG mapping.

Models currently used:
  iJN678     -> GCF_000009725 Synechocystis sp. PCC 6803      (Nogales 2012, BiGG SBML)
  iSynCJ816  -> GCF_000009725 Synechocystis sp. PCC 6803      (Joshi 2017, BiGG JSON)
  iJB785     -> GCF_000012525 Synechococcus elongatus PCC 7942 (Broddrick 2016, BiGG SBML)

Conditions:
  light : photon uptake allowed (default for iJB785; LB=-100 for iJN678);
          CO2/HCO3 uptake allowed (default).
  dark  : photon exchanges forced to 0; ATP can come from glucose if a
          glucose exchange exists. We do NOT enable a heterotrophic carbon
          source by default — so dark FBA may give 0 growth and primarily
          maintenance fluxes. This mirrors Knoop 2013's diel approach where
          the "dark" snapshot includes intracellular storage degradation.

For each reaction we report:
  rID, Reaction, EC (from notes), Pathway (from notes),
  Gene_Rule (full GPR), Genes (split list),
  Flux_Light, Flux_Dark, Flux_Diff, Flux_Amplitude.

Output: 03.flux-raw/{GCF}_{model}_native.tsv (one row per reaction-gene mapping).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

import cobra

ROOT = Path("/home/yangyicheng/27.evolution/39.flux_CR")
MODEL_DIR = ROOT / "08.models"
OUT_DIR = ROOT / "03.flux-raw"
MAP_DIR = ROOT / "04.merged"

# Model registry: photon_pattern is the substring matching photon exchange rxns;
# default_photon_lb is the per-reaction LB set in light mode; close_uptakes lists
# heterotrophic carbon uptakes to zero out under autotrophic light/dark conditions.
MODELS = {
    "iJN678":   {
        "gcf": "GCF_000009725",
        "dataset": "iJN678_lightdark",
        "format": "sbml",
        "photon_pat": "EX_photon_e",
        "photon_lb": -100.0,
        "close_uptakes": ["EX_glc__D_e", "EX_glyc_e", "EX_glcn__D_e", "EX_pyr_e", "EX_succ_e", "EX_ac_e"],
        "open_uptakes": [("EX_co2_e", -100.0), ("EX_hco3_e", -100.0)],
        "biomass_rxn": None,  # already set in SBML
    },
    "iSynCJ816": {
        "gcf": "GCF_000009725",
        "dataset": "iSynCJ816_lightdark",
        "format": "json",
        # iSynCJ816 has PHOTON_E1/680/700 as INTERNAL reactions (split photon pool
        # into wavelengths) -- we must NOT change their LB. Only the exchange
        # reaction EX_photon_e gates light uptake.
        "photon_pat": "EX_photon_e",
        "photon_lb": -100.0,
        "close_uptakes": ["EX_glc__D_e", "EX_glcglyc_e"],
        "open_uptakes": [("EX_co2_e", -100.0), ("EX_hco3_e", -100.0)],
        "biomass_rxn": "BIOMASS_Ec_SynAuto_1",  # autotrophic biomass; JSON has no objective set
    },
    "iJB785":   {
        "gcf": "GCF_000012525",
        "dataset": "iJB785_lightdark",
        "format": "sbml",
        "photon_pat": "EX_photon",
        "photon_lb": -1000.0,
        "close_uptakes": [],
        "open_uptakes": [],
        "biomass_rxn": None,
    },
    "PCC7002_Saha2016": {
        "gcf": "GCF_000019485",
        "dataset": "PCC7002_Saha2016_lightdark",
        "format": "json",
        "model_file": "PCC7002_model.json",
        # Photon already opened in default bounds; we keep it at -100 for light.
        "photon_pat": "EX_PHOTON_E",
        "photon_lb": -100.0,
        "close_uptakes": ["EX_GLC_E", "EX_GLC__D_E", "EX_GLYC_E", "EX_PYR_E", "EX_AC_E", "EX_SUCC_E"],
        "open_uptakes": [("EX_CO2_E", -100.0), ("EX_HCO3_E", -100.0)],
        # The Saha 2016 model has 4 biomass equations. BIOMASSCLIMITED (carbon-
        # limited, the original C-limited eq) gives ρ ≈ 0 against CR — its
        # stoichiometry forces ~30 housekeeping OGs to a tied minimum flux
        # (∼0.019 mmol·gDW⁻¹·h⁻¹), which lacks rank information vs CR.
        # NEWBIOMASSCLIMITED is the refined/updated version which gives a
        # clean ρ ≈ +0.28 (p < 0.05), consistent with the other 5 GEMs.
        "biomass_rxn": "NEWBIOMASSCLIMITED",
        # Model uses short locus IDs (A0001, B0123 ...); prepend prefix to match GFF old_locus_tag.
        "gene_prefix": "SYNPCC7002_",
    },
    "iSP1101_NIES39": {
        # Yoshikawa 2015 PLOS ONE — Arthrospira platensis NIES-39 (745 rxns, 621 genes)
        # Model built from XLSX supplementary by 08.models/build_iSP1101_from_xlsx.py
        # NIES-39 is a sister strain of Limnospira platensis C1 (GCF_025200965, in our
        # panel) — we attribute the projected flux to C1.
        "gcf": "GCF_025200965",
        "dataset": "iSP1101_NIES39_lightdark",
        "format": "json",
        "model_file": "iSP1101.json",
        "photon_pat": "AP739",                # AP739: photon[e] <=>
        "photon_lb": -100.0,
        "close_uptakes": ["AP724"],           # glc-D[e] exchange
        "open_uptakes": [("AP719", -100.0),   # co2[e]
                         ("AP729", -100.0)],  # hco3[e] (UB stays 1000)
        "biomass_rxn": "AP001",
        # Custom id_map: NIES39 locus -> WP_ -> OG (built by assign_og_NIES39.sh)
        "id_map_file": "NIES39_locus2og.tsv",
        "id_map_cols": {"gene": "locus_tag", "protein": "protein_id", "og": "OG"},
    },
    "iTeryR_GardnerBoyle2017": {
        # Gardner & Boyle 2017 BMC Syst Biol — iTri101 / iTeryR
        "gcf": "GCF_000014265",
        "dataset": "iTeryR_GardnerBoyle2017_lightdark",
        "format": "sbml",
        "model_file": "iTeryR_GardnerBoyle2017.xml",
        "photon_pat": "EX_photon",          # single photon exchange
        "photon_lb": -100.0,
        # Default model has EX_co2 fixed at -100/-100 (infeasible together with
        # other constraints); relax to allow 0..-100 CO2 uptake.
        "close_uptakes": [],
        "open_uptakes": [("EX_co2", -100.0)],   # also forces UB to 0 via patching below
        "biomass_rxn": None,                 # EX_biomass already set as objective
        "gene_prefix": "",                   # Tery_XXXX matches GFF old_locus_tag directly
        # Extra ad-hoc bound overrides applied verbatim before FBA:
        "bound_overrides": [
            ("EX_co2",  -100.0,    0.0),     # relax upper bound (was -100)
            ("EX_h2o", -1000.0, 1000.0),
            ("EX_fe2", -1000.0, 1000.0),
            ("EX_fe3", -1000.0, 1000.0),
        ],
    },
}


def load_model(model_name: str, cfg: dict):
    fmt = cfg.get("format", "sbml")
    custom = cfg.get("model_file")  # explicit override (e.g. PCC7002_model.json)
    if custom:
        path = MODEL_DIR / custom
    elif fmt == "json":
        path = MODEL_DIR / f"{model_name}.json"
    else:
        path = MODEL_DIR / f"{model_name}.xml"
    if fmt == "json":
        return cobra.io.load_json_model(str(path))
    return cobra.io.read_sbml_model(str(path))


def split_gpr(gpr: str):
    """Return unique gene IDs from a GPR rule string."""
    if not gpr:
        return []
    gpr = gpr.replace("(", " ").replace(")", " ")
    parts = re.split(r"\b(?:and|or)\b|\s+", gpr, flags=re.IGNORECASE)
    return sorted({p.strip() for p in parts if p.strip() and p.strip() not in {"and", "or"}})


def load_gene_to_protein(gcf: str, custom_map: str = None, col_map: dict = None):
    """For one strain, build a lookup: gene_id -> protein_id (WP_) and protein_id -> OG.

    Default loads `04.merged/{gcf}_id_map.tsv` and uses standard column names.
    If `custom_map` is given (filename in 04.merged), uses it instead with the
    column mapping in `col_map` = {"gene": ..., "protein": ..., "og": ...}.
    """
    g2p = {}
    p2og = {}
    if custom_map:
        p = MAP_DIR / custom_map
        gcol = col_map.get("gene")
        pcol = col_map.get("protein")
        ocol = col_map.get("og")
        with p.open() as f:
            rdr = csv.DictReader(f, delimiter="\t")
            for r in rdr:
                gene = r.get(gcol, "")
                pid  = r.get(pcol, "")
                og   = r.get(ocol, "")
                if gene and pid:
                    g2p.setdefault(gene, pid)
                if pid and og:
                    p2og[pid] = og
        return g2p, p2og

    p = MAP_DIR / f"{gcf}_id_map.tsv"
    with p.open() as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            pid = r["protein_id"]
            og = r["OG"]
            for col in ("gene_name", "locus_tag", "old_locus_tag", "protein_id"):
                v = r[col]
                if v and pid:
                    g2p.setdefault(v, pid)
            if pid and og:
                p2og[pid] = og
    return g2p, p2og


def configure_light(model: cobra.Model, photon_pat: str, photon_lb: float,
                    close_uptakes, open_uptakes, mode: str, bound_overrides=None):
    """Configure for autotrophic light or dark. Closes given heterotrophic uptakes
    in BOTH modes (always autotrophic) and opens given inorganic carbon uptakes.

    `bound_overrides` is an optional list of (rxn_id, lb, ub) tuples applied
    verbatim (used for models like iTeryR where the default bounds make FBA
    infeasible)."""
    rxn_ids = {r.id for r in model.reactions}
    for rid in close_uptakes:
        if rid in rxn_ids:
            model.reactions.get_by_id(rid).lower_bound = 0.0
    for rid, lb in open_uptakes:
        if rid in rxn_ids:
            r = model.reactions.get_by_id(rid)
            # Set bound to exactly lb (was: only loosen). Uniform across models
            # for cross-model flux comparability.
            r.lower_bound = lb
    if bound_overrides:
        for rid, lb, ub in bound_overrides:
            if rid in rxn_ids:
                r = model.reactions.get_by_id(rid)
                r.lower_bound, r.upper_bound = lb, ub
    # Only touch EXCHANGE reactions matching the photon pattern to avoid
    # accidentally flipping the LB of internal photon-splitting reactions
    # (e.g. PHOTON_E1, PHOTON680, PHOTON700 in iSynCJ816).
    # In light mode, force photon LB to photon_lb (cap any default that
    # exceeds it, e.g. PCC 7002 default -1000 -> -100, so flux scales are
    # comparable across models). In dark, force photon to 0.
    exchange_ids = {r.id for r in model.exchanges}
    for r in model.reactions:
        if photon_pat in r.id and r.id in exchange_ids:
            if mode == "light":
                r.lower_bound = photon_lb
            elif mode == "dark":
                r.lower_bound = 0.0
                r.upper_bound = max(0.0, r.upper_bound)


def run_one(model_name: str, cfg: dict):
    gcf = cfg["gcf"]; dataset = cfg["dataset"]
    photon_pat = cfg["photon_pat"]; photon_lb = cfg["photon_lb"]
    close_uptakes = cfg["close_uptakes"]
    biomass_rxn = cfg.get("biomass_rxn")
    gene_prefix = cfg.get("gene_prefix", "")
    print(f"\n=== {model_name} ({gcf}, {cfg.get('format','sbml')}) ===", file=sys.stderr)
    g2p, p2og = load_gene_to_protein(gcf, cfg.get("id_map_file"), cfg.get("id_map_cols"))
    print(f"  id_map: {len(g2p)} gene-ids, {len(p2og)} pid-with-OG", file=sys.stderr)
    if gene_prefix:
        print(f"  gene_prefix: '{gene_prefix}' will be tried on lookup", file=sys.stderr)

    fluxes = {}
    def _add_glycogen_sink(model, glyc_lb):
        """方案 C: 给 glycogen metabolite 加一个 sink reaction (虚拟内部储能源)."""
        mets_id = {m.id for m in model.metabolites}
        glyc_candidates = ["glycogen_c","glygn_c","m_glycogen_c","glycog_c",
                           "glycogen[c]","glycogen","M_glycogen_c","Glycogen_c"]
        for mid in glyc_candidates:
            if mid in mets_id:
                met = model.metabolites.get_by_id(mid)
                rxn_id = f"SK_glycogen_PLANC"
                if rxn_id in {r.id for r in model.reactions}:
                    rx = model.reactions.get_by_id(rxn_id)
                    rx.lower_bound = glyc_lb; rx.upper_bound = 0
                else:
                    new_rx = cobra.Reaction(rxn_id)
                    new_rx.name = "Glycogen sink (dark FBA, plan C)"
                    new_rx.lower_bound = glyc_lb; new_rx.upper_bound = 0
                    new_rx.add_metabolites({met: -1})
                    model.add_reactions([new_rx])
                return (mid, glyc_lb)
        return None

    def _force_atpm(model, atpm_lb):
        """方案 A: dark mode 强制 ATPM 下界. 返回 (forced_id, lb) 或 None."""
        rxn_ids = {r.id for r in model.reactions}
        atpm_candidates = ["ATPM","NGAM","ATP_M","ATPM_c","ATPM_C","AP_NGAM","ATPmaint",
                           "rxn00062","AP511","EN_ATP"]
        for aid in atpm_candidates:
            if aid in rxn_ids:
                rx = model.reactions.get_by_id(aid)
                rx.lower_bound = max(rx.lower_bound, atpm_lb)
                return (aid, rx.lower_bound)
        for r in model.reactions:
            if "maintenance" in (r.name or "").lower() and "atp" in (r.name or "").lower():
                r.lower_bound = max(r.lower_bound, atpm_lb)
                return (r.id, r.lower_bound)
        # 终极后备 (iJN678): 创建虚拟 ATPM
        mets = {m.id: m for m in model.metabolites}
        def find_met(prefixes):
            for p in prefixes:
                for mid in mets:
                    if mid.lower() == p or mid.lower().startswith(p + "_"):
                        return mets[mid]
            return None
        atp = find_met(["atp_c","atp","m_atp_c","cpd00002_c0"])
        h2o = find_met(["h2o_c","h2o","m_h2o_c","cpd00001_c0"])
        adp = find_met(["adp_c","adp","m_adp_c","cpd00008_c0"])
        pi  = find_met(["pi_c","pi","m_pi_c","cpd00009_c0"])
        hp  = find_met(["h_c","h","m_h_c","cpd00067_c0"])
        if all(x is not None for x in [atp, h2o, adp, pi, hp]):
            new_rx = cobra.Reaction("ATPM_FORCED")
            new_rx.name = "ATP maintenance (forced for dark FBA)"
            new_rx.lower_bound = atpm_lb; new_rx.upper_bound = 1000
            new_rx.add_metabolites({atp:-1, h2o:-1, adp:1, hp:1, pi:1})
            model.add_reactions([new_rx])
            return ("ATPM_FORCED (added)", atpm_lb)
        return None

    for mode in ("light", "dark"):
        model = load_model(model_name, cfg)
        if biomass_rxn:
            model.objective = biomass_rxn
        configure_light(model, photon_pat, photon_lb, close_uptakes, cfg["open_uptakes"], mode, cfg.get("bound_overrides"))

        # 方案 C: dark mode = ATPM 强制 + glycogen sink (虚拟内部储能源) + 自适应降级.
        # 模拟 Knoop 2013 的 "夜间细胞分解 glycogen 提供 ATP" 行为.
        # 三档配置尝试 (强→中→弱), 直到 feasible.
        if mode == "dark":
            # 尝试 3 档配置: (atpm_lb, glyc_lb)
            attempts = [(1.0, -5.0), (0.5, -10.0), (0.3, -20.0), (0.1, -50.0)]
            forced = None
            for atpm_lb, glyc_lb in attempts:
                # 重新 load + configure 干净环境
                model = load_model(model_name, cfg)
                if biomass_rxn: model.objective = biomass_rxn
                configure_light(model, photon_pat, photon_lb, close_uptakes, cfg["open_uptakes"],
                                mode, cfg.get("bound_overrides"))
                atpm = _force_atpm(model, atpm_lb)
                glyc = _add_glycogen_sink(model, glyc_lb)
                test_obj = model.slim_optimize()
                if np.isfinite(test_obj):
                    forced = (atpm, glyc, atpm_lb, glyc_lb)
                    print(f"  dark: ATPM={atpm[0] if atpm else 'none'}@lb={atpm_lb}  "
                          f"glyc_sink={glyc[0] if glyc else 'none'}@lb={glyc_lb}  ✓ feasible",
                          file=sys.stderr)
                    break
                else:
                    print(f"  dark: try (atpm={atpm_lb}, glyc={glyc_lb}) — infeasible", file=sys.stderr)
            if forced is None:
                # 全部尝试都 infeasible: 回退到 no-constraint dark
                model = load_model(model_name, cfg)
                if biomass_rxn: model.objective = biomass_rxn
                configure_light(model, photon_pat, photon_lb, close_uptakes, cfg["open_uptakes"],
                                mode, cfg.get("bound_overrides"))
                print(f"  dark: ALL plan-C attempts infeasible, fallback to no constraints", file=sys.stderr)

        obj = model.slim_optimize()
        print(f"  {mode}: max biomass obj={obj}", file=sys.stderr)
        try:
            sol = cobra.flux_analysis.pfba(model)
            print(f"    pFBA status={sol.status} tot_flux={sol.objective_value:.2f}", file=sys.stderr)
            fluxes[mode] = sol.fluxes
        except Exception as e:
            print(f"    pFBA failed ({e}); using plain FBA", file=sys.stderr)
            sol = model.optimize()
            print(f"    plain FBA status={sol.status} obj={sol.objective_value}", file=sys.stderr)
            fluxes[mode] = sol.fluxes

    # Build output rows
    model = load_model(model_name, cfg)
    if biomass_rxn:
        model.objective = biomass_rxn
    configure_light(model, photon_pat, photon_lb, close_uptakes, cfg["open_uptakes"], "light", cfg.get("bound_overrides"))
    out = OUT_DIR / f"{gcf}_{model_name}_native.tsv"
    cols = ["rID","Reaction","Subsystem","Gene_Rule","Gene","WP","OG",
            "Flux_Light","Flux_Dark","Flux_Diff","Flux_Amplitude",
            "Strain","Dataset"]
    n_rows, n_with_og, n_rxn = 0, 0, 0
    with out.open("w") as g:
        g.write("\t".join(cols) + "\n")
        for r in model.reactions:
            n_rxn += 1
            fl = float(fluxes["light"].get(r.id, 0.0))
            fd = float(fluxes["dark"].get(r.id, 0.0))
            diff = fl - fd
            amp = abs(diff)
            subsystem = r.subsystem or ""
            gpr = r.gene_reaction_rule or ""
            genes = split_gpr(gpr)
            if not genes:
                g.write("\t".join([r.id, r.name or "", subsystem, gpr, "", "", "",
                                   f"{fl}", f"{fd}", f"{diff}", f"{amp}",
                                   gcf, dataset]) + "\n")
                n_rows += 1
                continue
            for gene in genes:
                wp = g2p.get(gene, "")
                if not wp and gene_prefix:
                    wp = g2p.get(f"{gene_prefix}{gene}", "")
                if not wp and gene.startswith("WP_"):
                    wp = gene
                og = p2og.get(wp, "") if wp else ""
                g.write("\t".join([r.id, r.name or "", subsystem, gpr, gene, wp, og,
                                   f"{fl}", f"{fd}", f"{diff}", f"{amp}",
                                   gcf, dataset]) + "\n")
                n_rows += 1
                if og:
                    n_with_og += 1
    print(f"  wrote {n_rows} rows ({n_with_og} with OG) over {n_rxn} reactions -> {out}", file=sys.stderr)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in MODELS.items():
        run_one(name, cfg)


if __name__ == "__main__":
    main()
