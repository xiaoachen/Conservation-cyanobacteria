#!/usr/bin/env python3
"""Knoop 2013 真实昼夜 FBA × 扩展 CR 数据库 — 关键对照分析

数据源（CR 字典优先级，后者覆盖前者）:
  1. 28.mrcr-20250602/01.singlecopy_gene/    222 OG (by OG ID)
  2. 28.mrcr-20250602/02-11/                  67 gene (by gene name)
  3. 34.KaiABC/01.KaiABC, 02.clock-output, 03.PSII_D1  ~7 gene
  4. 35.metabolism/Lipid + Nucleotide...     92 gene
  5. 27.allophycocyanin/antenna_cr_summary.tsv  11 antenna gene

Knoop 反应 → CR 链:
  rID → PCC 6803 locus (sll/slr/sml/smr)
       ├─ via id_map.OG → CR (走 28.mrcr.01 OG ID 路径)
       └─ via id_map.gene_name → CR (走基因名路径, 命中 28.mrcr.02-11/34/35/27)

为什么这份数据关键:
  - 39 的 FBA: Dark > 0 仅 5.2% (Amp ≈ |L|, 不能区分)
  - Knoop:    Dark > 0 占 57.7% (Amp 与 |L| 真分得开, ρ(|L|,|D|)=0.46)

输出:
  knoop_per_og.tsv                         每个 (OG/gene) 的 amp/|L|/|D|/CR/source
  knoop_correlation_results.tsv            4 度量 + partial corr 表
  figures/01..04_knoop_*.{png,pdf}         4 张图
"""
from pathlib import Path
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, linregress, rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent          # 39.flux_CR
EVO = ROOT.parent           # 27.evolution
FIG = HERE / "figures"; FIG.mkdir(exist_ok=True)


# ============================================================
# 1. Build unified CR lookup from 5 sources
# ============================================================
def load_all_cr():
    """Return DataFrame[key, CR, source]. key 可以是 OG ID (OG0xxxx) 或 gene name (psbA)。
    多源命中同一 key 时, 后者覆盖前者 (顺序见下)。
    """
    records = []
    def _add(path, label, key_col="OG", cr_col="CR"):
        if not path.exists():
            print(f"  [SKIP] not found: {path}", file=sys.stderr); return
        d = pd.read_csv(path) if path.suffix == ".csv" else pd.read_csv(path, sep="\t")
        # 27.allophycocyanin tsv 用 CR_median
        if cr_col not in d.columns:
            for alt in ("CR_median", "CR_mean", "median_CR"):
                if alt in d.columns:
                    cr_col = alt; break
        if key_col not in d.columns:
            for alt in ("gene", "Gene", "og", "key"):
                if alt in d.columns:
                    key_col = alt; break
        d = d[[key_col, cr_col]].copy()
        d.columns = ["key", "CR"]
        d["CR"] = pd.to_numeric(d["CR"], errors="coerce")
        d = d.dropna(subset=["key","CR"])
        d = d[(d["CR"] >= 0) & (d["CR"] <= 1)]
        d["source"] = label
        n = len(d); print(f"  [{label:32s}] {n} entries", file=sys.stderr)
        records.append(d)

    # 1. singlecopy by OG ID
    _add(EVO / "28.mrcr-20250602/01.singlecopy_gene/01.singlecopy_gene_mr_cr_aggregated.csv",
         "28.mrcr/01.singlecopy_gene")
    # 2-11 by gene name
    for mod in ["02.Photosystem_II","03.Photosystem_I","04.CBB_cycle",
                "05.Carbon_concentrating_mechanism","06.ATP_synthase",
                "07.Cytochrome_b6f","08.Photoprotection","09.NDH",
                "10.Phycobilisome","11.Pigment_biosynthesis"]:
        _add(EVO / f"28.mrcr-20250602/{mod}/{mod}_mr_cr_aggregated.csv", f"28.mrcr/{mod}")
    # 34.KaiABC
    for sub in ["01.KaiABC","02.clock-output","03.PSII_D1"]:
        _add(EVO / f"34.KaiABC/{sub}/{sub}_mr_cr_aggregated.csv", f"34.KaiABC/{sub}")
    # 35.metabolism
    for sub in ["Lipid_Metabolism","Nucleotide_Carbohydrate_Amino_Acid_Energy_Metabolism"]:
        _add(EVO / f"35.metabolism/{sub}/Result/{sub}_mr_cr_aggregated.csv", f"35.metabolism/{sub}")
    # 27.allophycocyanin (用 antenna_cr_summary.tsv)
    _add(EVO / "27.allophycocyanin/antenna_cr_summary.tsv", "27.allophycocyanin")

    big = pd.concat(records, ignore_index=True)
    # Dedup, 保留最后一次 (= 最高优先级)
    big = big.drop_duplicates(subset="key", keep="last").reset_index(drop=True)
    print(f"\n  [TOTAL UNIQUE KEYS]: {len(big)}", file=sys.stderr)
    return big

print("[CR] 加载所有 CR 数据源:", file=sys.stderr)
cr_df = load_all_cr()
cr_map = dict(zip(cr_df["key"], cr_df["CR"]))
cr_src = dict(zip(cr_df["key"], cr_df["source"]))
n_og = sum(1 for k in cr_map if str(k).startswith("OG"))
n_gene = len(cr_map) - n_og
print(f"  → 总 keys: {len(cr_map)} (OG-ID: {n_og}, gene-name: {n_gene})", file=sys.stderr)


# ============================================================
# 2. Parse Knoop light_dark.csv
# ============================================================
ld = pd.read_csv(ROOT / "light_dark.csv", skiprows=2, header=None,
                 names=["FBA_L","MIN_L","MAX_L","FBA_D","MIN_D","MAX_D","Reaction"])
ld["rID"] = ld["Reaction"].str.extract(r"^(R\d+)")
ld = ld.dropna(subset=["rID"]).copy()
for c in ["FBA_L","FBA_D"]:
    ld[c] = pd.to_numeric(ld[c], errors="coerce")
ld["abs_L"] = ld["FBA_L"].abs()
ld["abs_D"] = ld["FBA_D"].abs()
ld["amp"]   = (ld["FBA_L"] - ld["FBA_D"]).abs()
ld["avg"]   = (ld["abs_L"] + ld["abs_D"]) / 2.0
print(f"\n[1] light_dark.csv -> {len(ld)} reactions  (Dark>0 = {100*(ld['abs_D']>0).mean():.1f}%)", file=sys.stderr)


# ============================================================
# 3. Parse pcbi.1003081.s007 (rID → genes)
# ============================================================
rxn = pd.read_csv(ROOT / "pcbi.1003081.s007.csv", encoding="latin-1")[["rID","genes","Pathway"]]
rxn["genes"] = rxn["genes"].astype(str)
def split_genes(s):
    if not s or s == "nan": return []
    parts = re.split(r",| or | and |\s*\+\s*", s)
    return [p.strip() for p in parts if re.match(r"^s[lm][lr]\d+$", p.strip())]
rxn["gene_list"] = rxn["genes"].apply(split_genes)
rxn = rxn[rxn["gene_list"].str.len() > 0].copy()
print(f"[2] rID → gene 映射 -> {len(rxn)} 反应有有效 PCC 6803 locus", file=sys.stderr)


# ============================================================
# 4. id_map: locus → (OG ID, gene_name)
# ============================================================
idmap = pd.read_csv(ROOT / "04.merged/GCF_000009725_id_map.tsv", sep="\t")
loc2og = dict(zip(idmap["old_locus_tag"], idmap["OG"]))
loc2gn = dict(zip(idmap["old_locus_tag"],
                  idmap["gene_name"].astype(str).where(
                      ~idmap["gene_name"].astype(str).str.startswith("SGL_RS"),
                      other=np.nan)))
# 把 psbA1, psbA2, psbA3 -> psbA 等数字后缀规范化（多基因家族同 OG/同 CR key）
def normalize_gene(g):
    if g is None or (isinstance(g, float) and pd.isna(g)): return None
    g = str(g)
    # 一些 PCC 6803 基因带后缀的标准化: psbA1->psbA, psaL/psaJ 不动
    m = re.match(r"^([a-z]{3,4})\d+$", g)
    return m.group(1) if m else g
print(f"[3] id_map -> {len(loc2og)} locus → OG; real gene names: {sum(1 for v in loc2gn.values() if isinstance(v, str))}", file=sys.stderr)


# ============================================================
# 5. Expand (rxn × gene) and lookup CR via (1) OG-ID then (2) gene-name
# ============================================================
rows = []
n_via_og = 0; n_via_gene = 0; n_miss = 0
for _, r in rxn.iterrows():
    for loc in r["gene_list"]:
        # Try 1: locus → OG → CR
        og = loc2og.get(loc)
        cr = None; key_used = None; src = None
        if og and og in cr_map:
            cr = cr_map[og]; key_used = og; src = cr_src[og]; n_via_og += 1
        else:
            # Try 2: locus → gene_name (raw + normalized) → CR
            gn_raw = loc2gn.get(loc)
            if isinstance(gn_raw, str):
                gn_norm = normalize_gene(gn_raw)
                for cand in (gn_raw, gn_norm):
                    if cand in cr_map:
                        cr = cr_map[cand]; key_used = cand; src = cr_src[cand]; n_via_gene += 1; break
            if cr is None:
                n_miss += 1
        rows.append({"rID": r["rID"], "locus": loc,
                     "OG": og or "", "gene_name": loc2gn.get(loc, ""),
                     "key": key_used or "", "CR": cr, "src": src or "",
                     "Pathway": r["Pathway"]})

rg = pd.DataFrame(rows)
print(f"[4] expand (rxn × locus): {len(rg)} entries", file=sys.stderr)
print(f"    CR via OG-ID:      {n_via_og}", file=sys.stderr)
print(f"    CR via gene-name:  {n_via_gene}", file=sys.stderr)
print(f"    No CR:             {n_miss}  ({100*n_miss/len(rg):.1f}%)", file=sys.stderr)
print(f"    UNIQUE keys covered: {rg[rg['CR'].notna()]['key'].nunique()}", file=sys.stderr)


# ============================================================
# 6. Merge flux into (rxn × locus × key)
# ============================================================
flux = ld[["rID","abs_L","abs_D","amp","avg"]]
merged = rg.merge(flux, on="rID", how="inner").dropna(subset=["CR"])
print(f"[5] merged with flux + CR: {len(merged)} rows", file=sys.stderr)


# ============================================================
# 7. Aggregate to (key)-level
# ============================================================
og_tab = (merged.groupby("key", as_index=False)
          .agg(abs_L_max=("abs_L","max"),
               abs_D_max=("abs_D","max"),
               amp_max=("amp","max"),
               avg_max=("avg","max"),
               n_reactions=("rID","nunique"),
               n_loci=("locus","nunique"),
               CR=("CR","first"),
               src=("src","first"),
               Pathways=("Pathway", lambda x: ";".join(sorted(set(x.dropna().astype(str))))[:200])))
og_tab.to_csv(HERE / "knoop_per_og.tsv", sep="\t", index=False)
print(f"[6] OG/gene-level table: {len(og_tab)} entries with CR  →  knoop_per_og.tsv", file=sys.stderr)
print(f"    By source breakdown:")
print(og_tab.groupby("src").size().to_string())


# ============================================================
# 8. Correlation analysis — 4 metrics × {raw, partial}
# ============================================================
def fisher_ci(rho, n):
    if n < 4 or not np.isfinite(rho) or abs(rho) >= 1: return (np.nan, np.nan)
    z = np.arctanh(rho); se = 1.0/np.sqrt(n-3)
    return np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)

def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([rz, np.ones_like(rz)])
    bx, *_ = np.linalg.lstsq(Z, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Z, ry, rcond=None)
    ex, ey = rx - Z @ bx, ry - Z @ by
    return spearmanr(ex, ey)

results = []
for metric, col in [("Amp","amp_max"), ("|L|","abs_L_max"),
                    ("|D|","abs_D_max"), ("Avg","avg_max")]:
    sub = og_tab[og_tab[col] > 0]
    n = len(sub)
    rho, p = (spearmanr(np.log2(sub[col]), sub["CR"]) if n >= 5 else (np.nan, np.nan))
    lo, hi = fisher_ci(rho, n)
    results.append({"metric": metric, "n": n, "rho": rho, "p": p, "ci_lo": lo, "ci_hi": hi, "note": ""})

print("\n[7] (key)-level Spearman ρ — log₂(metric) vs CR:")
print(pd.DataFrame(results).round(4).to_string(index=False))

both = og_tab[(og_tab["amp_max"] > 0) & (og_tab["abs_L_max"] > 0)].copy()
both["log_amp"] = np.log2(both["amp_max"])
both["log_absL"] = np.log2(both["abs_L_max"])
both_D = og_tab[og_tab["abs_D_max"] > 0].copy()
both_D["log_D"] = np.log2(both_D["abs_D_max"])

print(f"\n[8] Partial correlation (n={len(both)}, both Amp>0 & |L|>0):")
r_amp, p_amp = spearmanr(both["log_amp"], both["CR"])
r_L,   p_L   = spearmanr(both["log_absL"], both["CR"])
r_amp_g_L, p_amp_g_L = partial_spearman(both["log_amp"], both["CR"], both["log_absL"])
r_L_g_amp, p_L_g_amp = partial_spearman(both["log_absL"], both["CR"], both["log_amp"])
r_D, p_D = (spearmanr(both_D["log_D"], both_D["CR"]) if len(both_D) >= 5 else (np.nan, np.nan))
print(f"  Amp ~ CR (raw):        ρ = {r_amp:+.3f}  p = {p_amp:.3e}")
print(f"  |L| ~ CR (raw):        ρ = {r_L:+.3f}  p = {p_L:.3e}")
print(f"  Amp ~ CR | |L|:        ρ = {r_amp_g_L:+.3f}  p = {p_amp_g_L:.3e}")
print(f"  |L| ~ CR | Amp:        ρ = {r_L_g_amp:+.3f}  p = {p_L_g_amp:.3e}")
print(f"  |D| ~ CR:              ρ = {r_D:+.3f}  p = {p_D:.3e}  n={len(both_D)}")

results.extend([
    {"metric": "Amp | |L|  (partial)", "n": len(both), "rho": r_amp_g_L, "p": p_amp_g_L, "ci_lo": np.nan, "ci_hi": np.nan, "note": "partial corr"},
    {"metric": "|L| | Amp  (partial)", "n": len(both), "rho": r_L_g_amp, "p": p_L_g_amp, "ci_lo": np.nan, "ci_hi": np.nan, "note": "partial corr"},
])
pd.DataFrame(results).to_csv(HERE / "knoop_correlation_results.tsv", sep="\t", index=False)


# ============================================================
# 9. Plots
# ============================================================
METRIC_COLOR = {"Amp": "#444444", "|L|": "#FF9F1C", "Avg": "#2EC4B6", "|D|": "#5A3D78"}

# 9.1 4 metric bar
fig, ax = plt.subplots(figsize=(7, 4.7), constrained_layout=True)
rs = pd.DataFrame(results[:4]).set_index("metric")
x = np.arange(len(rs))
ax.bar(x, rs["rho"].fillna(0), color=[METRIC_COLOR[m] for m in rs.index],
       edgecolor="black", lw=0.7, alpha=0.85)
err_lo = (rs["rho"] - rs["ci_lo"]).fillna(0)
err_hi = (rs["ci_hi"] - rs["rho"]).fillna(0)
ax.errorbar(x, rs["rho"], yerr=[err_lo, err_hi], fmt="none", ecolor="black", capsize=4, lw=1.0)
for i, (m, r) in enumerate(rs.iterrows()):
    if not np.isfinite(r["rho"]):
        ax.text(i, 0.01, "n/a", ha="center", fontsize=10, color="#888"); continue
    sig = "***" if r["p"] < 1e-3 else ("**" if r["p"] < 1e-2 else ("*" if r["p"] < 0.05 else ""))
    ax.text(i, r["rho"] + (0.015 if r["rho"] >= 0 else -0.03),
            f"{r['rho']:+.3f}{sig}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i, ax.get_ylim()[0] + 0.01, f"n={int(r['n'])}", ha="center", fontsize=8, color="#444")
ax.axhline(0, color="#888", lw=0.5)
ax.set_xticks(x); ax.set_xticklabels(rs.index, fontsize=11)
ax.set_ylabel("Spearman's ρ  (log₂(metric) vs CR)", fontsize=10)
ax.set_title(f"Knoop 2013 real diurnal FBA × extended CR DB — 4 flux metrics  (n={len(og_tab)})",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(FIG / "01_knoop_4metric_bar.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "01_knoop_4metric_bar.pdf", bbox_inches="tight")
plt.close(fig); print("[OK] 01_knoop_4metric_bar")


# 9.2 Partial correlation forest
fig, ax = plt.subplots(figsize=(8.5, 4), constrained_layout=True)
labels = [
    "log Amp ~ CR  (raw)",
    "log |L|  ~ CR  (raw)",
    "log Amp ~ CR | log |L|  (partial)",
    "log |L|  ~ CR | log Amp  (partial)",
]
vals = [r_amp, r_L, r_amp_g_L, r_L_g_amp]
pvals = [p_amp, p_L, p_amp_g_L, p_L_g_amp]
colors = ["#444444", "#FF9F1C", "#999999", "#FFC380"]
y = np.arange(len(labels))[::-1]
ax.scatter(vals, y, s=140, c=colors, edgecolor="black", zorder=3)
for yi, v, p in zip(y, vals, pvals):
    sig = "***" if p < 1e-3 else ("**" if p < 1e-2 else ("*" if p < 0.05 else "n.s."))
    ax.text(v + 0.012, yi, f"ρ={v:+.3f}  ({sig})", va="center", fontsize=10)
ax.axvline(0, color="#888", lw=0.5, ls="--")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Spearman's ρ vs CR", fontsize=10)
ax.set_title(f"Amp vs |L| independent contribution — Partial correlation (n = {len(both)})",
             fontsize=11, fontweight="bold")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(FIG / "02_knoop_partial_corr.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "02_knoop_partial_corr.pdf", bbox_inches="tight")
plt.close(fig); print("[OK] 02_knoop_partial_corr")


# 9.3 Amp vs |L| scatter (颜色 = CR)
fig, ax = plt.subplots(figsize=(8, 6.5), constrained_layout=True)
sc = ax.scatter(both["log_absL"], both["log_amp"], c=both["CR"],
                cmap="viridis", vmin=0, vmax=1, s=60, alpha=0.85,
                edgecolor="white", linewidth=0.5)
lo = min(both["log_absL"].min(), both["log_amp"].min())
hi = max(both["log_absL"].max(), both["log_amp"].max())
ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1.0, label="y = x")
ax.set_xlabel("log₂(|Flux_Light|)", fontsize=11)
ax.set_ylabel("log₂(Flux Amplitude) = log₂|L − D|", fontsize=11)
ax.set_title(f"Knoop 2013 (key-level): Amp vs |L|  (color = CR; n={len(both)})",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.colorbar(sc, ax=ax, label="CR", shrink=0.8)
fig.savefig(FIG / "03_knoop_amp_vs_absL.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "03_knoop_amp_vs_absL.pdf", bbox_inches="tight")
plt.close(fig); print("[OK] 03_knoop_amp_vs_absL")


# 9.4 Main scatter: log₂(Amp) × CR with Dark-dominant highlighted + source colored
fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
plot_d = og_tab[og_tab["amp_max"] > 0].copy()
plot_d["log_amp"] = np.log2(plot_d["amp_max"])
plot_d["dark_dominant"] = plot_d["abs_D_max"] > plot_d["abs_L_max"]
# 颜色按 source 简化分类
def source_color(s):
    if "singlecopy" in s: return "#888888"
    if "Photosystem_II" in s or "Photosystem_I" in s: return "#3A7BCF"
    if "CBB" in s or "metabolism" in s or "Cytochrome" in s or "NDH" in s or "ATP" in s: return "#2EC4B6"
    if "KaiABC" in s or "clock" in s: return "#E14B58"
    if "allophyco" in s or "Phycobil" in s: return "#9467BD"
    if "Pigment" in s or "Carbon" in s or "Photopro" in s: return "#FF9F43"
    return "#444444"
plot_d["color"] = plot_d["src"].apply(source_color)

for src, sub in plot_d.groupby("src"):
    c = source_color(src)
    ax.scatter(sub["log_amp"], sub["CR"], s=55, color=c,
               alpha=0.7, edgecolor="white", linewidth=0.4, label=src.split("/")[-1])

# Dark-dominant 边框
dd = plot_d[plot_d["dark_dominant"]]
ax.scatter(dd["log_amp"], dd["CR"], s=130, facecolor="none",
           edgecolor="red", linewidth=1.2, label=f"Dark-dominant (n={len(dd)})")

lr = linregress(plot_d["log_amp"], plot_d["CR"])
xs = np.linspace(plot_d["log_amp"].min(), plot_d["log_amp"].max(), 100)
ax.plot(xs, lr.intercept + lr.slope*xs, color="black", lw=1.5, alpha=0.8)
rho, p = spearmanr(plot_d["log_amp"], plot_d["CR"])
ax.text(0.04, 0.06,
        f"Spearman ρ = {rho:+.3f}\nN = {len(plot_d)}\nP = {p:.2e}\nKnoop 2013 + ext. CR DB",
        transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="0.7"))
ax.set_xlabel("log₂(Flux Amplitude)  =  log₂|Flux_Light − Flux_Dark|", fontsize=11)
ax.set_ylabel("Conservation Ratio (CR)", fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title(f"Knoop 2013 diurnal FBA × extended CR  (n={len(plot_d)} keys; CR from 5 sources)",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=7, frameon=True, ncol=2)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
fig.savefig(FIG / "04_knoop_main_scatter.png", dpi=300, bbox_inches="tight")
fig.savefig(FIG / "04_knoop_main_scatter.pdf", bbox_inches="tight")
plt.close(fig); print("[OK] 04_knoop_main_scatter")

print("\nDone.")
