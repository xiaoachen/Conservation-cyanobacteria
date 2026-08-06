#!/usr/bin/env python3
"""Supplementary Figure S1d/S1e/S1f — model confidence and taxon sampling (supports R1).

S1d  Residue-level pLDDT distribution over every AlphaFold3 model predicted in this
     study, with the per-model mean distribution inset.  R1 quotes a mean pLDDT and a
     fraction of residues above 70 but the manuscript currently shows neither.
S1e  The same confidence resolved per strain, to show that model quality does not vary
     systematically across the phylogenetic and ecological breadth of the panel.
S1f  The sampling itself: genome size of every strain, with habitat and diazotrophy
     marked, supporting R1's claim that the panel spans the breadth of the clade.

Palette follows Figure 2: genome blue #3A7BCF, ecology pink #D16BA5, black reference
lines at lw 1.2.

Inputs : plddt_per_model.csv, plddt_residue_hist.csv  (this directory; produced by the
             cif traversal, see README note at the bottom of the module docstring)
         39.flux_CR/02.metadata/strain_meta.tsv
         13.genomesize_cr/cr_table.csv
         29.Nitrogen_fixation/list, list1   (diazotroph / non-diazotroph)
         30.fresh_water/list, list1         (freshwater / marine)
Outputs: FigureS1d_plddt_distribution.{pdf,png}
         FigureS1e_plddt_by_strain.{pdf,png}
         FigureS1f_strain_sampling.{pdf,png}
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("/home/yangyicheng/27.evolution")
OUT = Path(__file__).resolve().parent

C_BLUE = "#3A7BCF"      # Fig 2c,d
C_PINK = "#D16BA5"      # Fig 2a,b
C_GREY = "#8C8C8C"
PLDDT_CUT = 70.0
OUTGROUP = "GCF_003149375"          # Vampirovibrio chlorellavorus

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 1.0, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})


def load_meta():
    meta = pd.read_csv(ROOT / "39.flux_CR/02.metadata/strain_meta.tsv", sep="\t")
    meta["GCF"] = meta.GCF.str.replace(r"\.\d+$", "", regex=True)
    gs = pd.read_csv(ROOT / "13.genomesize_cr/cr_table.csv",
                     usecols=["GCF", "genome_size", "protein_num"])
    gs["GCF"] = gs.GCF.str.replace(r"\.\d+$", "", regex=True)
    meta = meta.merge(gs, on="GCF", how="left")

    def read_list(p):
        return {l.strip().split(".")[0] for l in open(ROOT / p) if l.strip()}

    fix = read_list("29.Nitrogen_fixation/list")
    fresh = read_list("30.fresh_water/list")
    meta["diazotroph"] = meta.GCF.isin(fix)
    meta["freshwater"] = meta.GCF.isin(fresh)
    meta["label"] = meta.Organism.str.replace(r"\s*=.*$", "", regex=True)
    return meta


def load_models(meta):
    """Per-model table restricted to the declared 36-genome panel.

    The traversal finds 12,903 unique models over 37 assemblies; 278 of them belong to
    GCF_000011545, which is absent from strain_meta.tsv.  Restricting to the declared
    panel changes the mean pLDDT by 0.01, but keeps the reported N defensible.
    """
    m = pd.read_csv(OUT / "plddt_per_model.csv")
    n_all = len(m)
    m = m[m.gcf.isin(set(meta.GCF))].copy()
    print(f"models: {n_all:,} traversed -> {len(m):,} inside the 36-genome panel "
          f"({n_all - len(m):,} dropped, non-panel assemblies)")
    return m


def panel_d(m):
    # histogram and quoted N both refer to the 36-genome panel (12,625 models)
    h = pd.read_csv(OUT / "plddt_residue_hist_panel.csv")   # panel-restricted
    centre = h.plddt_bin + 0.5
    tot = h.n_residues.sum()
    mean_res = float((h.n_residues * centre).sum() / tot)
    frac70 = float(h.loc[h.plddt_bin >= PLDDT_CUT, "n_residues"].sum()) / tot

    fig = plt.figure(figsize=(5.2, 3.3))
    ax = fig.add_axes([0.135, 0.180, 0.835, 0.735])
    lo = h.plddt_bin < PLDDT_CUT
    ax.bar(h.plddt_bin[lo] + 0.5, h.n_residues[lo] / 1e3, width=1.0,
           color=C_GREY, alpha=0.85, linewidth=0)
    ax.bar(h.plddt_bin[~lo] + 0.5, h.n_residues[~lo] / 1e3, width=1.0,
           color=C_BLUE, alpha=0.85, linewidth=0)
    ax.axvline(PLDDT_CUT, color="black", lw=1.2, ls="--", alpha=0.8)
    ax.axvline(mean_res, color=C_PINK, lw=1.4)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Per-residue pLDDT")
    ax.set_ylabel("Residues (thousands)")
    ax.set_title("Prediction confidence across the 36-genome panel",
                 fontsize=8.5, loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.03, 0.96,
            f"{len(m):,} models · {tot:,} residues\n"
            f"mean pLDDT = {mean_res:.1f}\n"
            f"{frac70*100:.1f}% of residues ≥ {PLDDT_CUT:.0f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", lw=0.6))
    ax.text(PLDDT_CUT - 1.5, ax.get_ylim()[1] * 0.78, "analysis cutoff",
            rotation=90, ha="right", va="center", fontsize=6.4, color="0.3")

    inset = ax.inset_axes([0.155, 0.34, 0.28, 0.34])
    inset.hist(m.mean_plddt, bins=40, color=C_BLUE, alpha=0.85, linewidth=0)
    inset.axvline(m.mean_plddt.mean(), color=C_PINK, lw=1.2)
    inset.set_xlabel("per-model mean", fontsize=6)
    inset.set_ylabel("models", fontsize=6, labelpad=1)
    inset.tick_params(labelsize=5.5)
    inset.spines[["top", "right"]].set_visible(False)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigureS1d_plddt_distribution.{ext}", dpi=600)
    plt.close(fig)
    return len(m), tot, mean_res, frac70


def panel_e(meta, m):
    m = m.merge(meta[["GCF", "label"]], left_on="gcf", right_on="GCF", how="left")
    m["label"] = m.label.fillna(m.gcf)
    stat = (m.groupby("label").mean_plddt
             .agg(med="median", n="size").sort_values("med"))
    keep = stat[stat.n >= 5]

    fig = plt.figure(figsize=(6.4, 6.9))
    ax = fig.add_axes([0.475, 0.135, 0.400, 0.825])
    y = np.arange(len(keep))
    data = [m.loc[m.label == lab, "mean_plddt"].to_numpy() for lab in keep.index]
    bp = ax.boxplot(data, positions=y, vert=False, widths=0.66,
                    patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor=C_BLUE, edgecolor="black", linewidth=0.7, alpha=0.75)
    for k in ("whiskers", "caps"):
        for a in bp[k]:
            a.set(color="black", linewidth=0.7)
    for a in bp["medians"]:
        a.set(color="black", linewidth=1.3)

    # both reference lines sit ABOVE the boxes, otherwise the grand-mean line runs
    # straight through them and becomes invisible
    gmean = m.mean_plddt.mean()
    ax.axvline(gmean, color=C_PINK, lw=1.6, zorder=5)
    ax.axvline(PLDDT_CUT, color="black", lw=1.2, ls="--", alpha=0.85, zorder=5)
    ax.annotate(f"grand mean {gmean:.1f}", xy=(gmean, len(keep) - 0.5),
                xytext=(3, 0), textcoords="offset points", color=C_PINK,
                fontsize=6.4, va="center", ha="left", fontweight="bold")
    ax.annotate(f"pLDDT ≥ {PLDDT_CUT:.0f} cutoff", xy=(PLDDT_CUT, len(keep) - 0.5),
                xytext=(3, 0), textcoords="offset points", color="0.25",
                fontsize=6.4, va="center", ha="left")

    ax.set_yticks(y)
    ax.set_yticklabels(keep.index, fontsize=6.4, style="italic")
    ax.set_ylim(-0.8, len(keep) + 0.4)
    ax.set_xlim(66, 100.5)
    ax.set_xlabel("Per-model mean pLDDT")
    ax.set_title("Model confidence is uniform across the panel", fontsize=8.5,
                 loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, lab in zip(y, keep.index):
        ax.text(1.02, yi, f"n={int(keep.loc[lab,'n']):<4d}{keep.loc[lab,'med']:.1f}",
                transform=tr, va="center", ha="left", fontsize=5.8, color="0.35",
                clip_on=False, family="DejaVu Sans Mono")
    fig.text(0.475, 0.018,
             "One box per strain, over that strain's per-model mean pLDDT.\n"
             "The grand mean sits below almost every strain median because the\n"
             f"per-model distribution is left-skewed (mean {gmean:.1f}, median "
             f"{m.mean_plddt.median():.1f}).",
             fontsize=6.2, va="bottom", ha="left", color="0.25")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigureS1e_plddt_by_strain.{ext}", dpi=600)
    plt.close(fig)
    return keep, gmean


def panel_f(meta):
    d = meta.dropna(subset=["genome_size"]).copy()
    d["Mb"] = d.genome_size / 1e6
    d = d.sort_values("Mb")
    fig = plt.figure(figsize=(7.2, 6.8))
    ax = fig.add_axes([0.430, 0.125, 0.360, 0.830])
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        col = C_BLUE if r.freshwater else C_PINK
        if r.GCF == OUTGROUP:
            col = "0.35"
        ax.plot([0, r.Mb], [yi, yi], color="0.85", lw=0.8, zorder=1)
        ax.scatter(r.Mb, yi, s=54, marker="^" if r.diazotroph else "o",
                   facecolor=col, edgecolor="black", linewidth=0.7, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(d.label, fontsize=6.4, style="italic")
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlim(0, d.Mb.max() * 1.08)
    ax.set_xlabel("Genome size (Mb)")
    ax.set_title("Genome size and habitat across the 36-genome panel",
                 fontsize=8.5, loc="left", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yi, (_, r) in zip(y, d.iterrows()):
        pn = "" if pd.isna(r.protein_num) else f"{int(r.protein_num):>5d}"
        ax.text(1.02, yi, f"{r.Mb:5.2f} Mb {pn}", transform=tr, va="center",
                ha="left", fontsize=5.8, color="0.35", clip_on=False,
                family="DejaVu Sans Mono")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markerfacecolor=C_BLUE,
               markeredgecolor="black", markersize=6, label="freshwater"),
        Line2D([], [], marker="o", ls="", markerfacecolor=C_PINK,
               markeredgecolor="black", markersize=6, label="non-freshwater"),
        Line2D([], [], marker="^", ls="", markerfacecolor="white",
               markeredgecolor="black", markersize=6, label="diazotroph"),
        Line2D([], [], marker="o", ls="", markerfacecolor="0.35",
               markeredgecolor="black", markersize=6, label="outgroup"),
    ], fontsize=6.2, loc="lower right", frameon=True, edgecolor="0.7",
        handletextpad=0.3, borderpad=0.4)

    fig.text(0.430, 0.016,
             f"Genome size spans {d.Mb.min():.2f}–{d.Mb.max():.2f} Mb, "
             f"proteome size {int(d.protein_num.min()):,}–"
             f"{int(d.protein_num.max()):,} proteins.\n"
             f"{int(d.diazotroph.sum())} diazotrophs; "
             f"{int(d.freshwater.sum())} strains are annotated freshwater and the "
             "remaining\n"
             f"{len(d) - int(d.freshwater.sum())} are pooled as non-freshwater "
             "(marine, brackish, soil, thermal and symbiotic).\n"
             f"{len(d)} of the 36 panel genomes carry genome-size metadata in the "
             "current table.",
             fontsize=6.2, va="bottom", ha="left", color="0.25")

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"FigureS1f_strain_sampling.{ext}", dpi=600)
    plt.close(fig)
    return d


def main():
    meta = load_meta()
    models = load_models(meta)
    n_mod, n_res, mean_res, frac70 = panel_d(models)
    keep, gmean = panel_e(meta, models)
    d = panel_f(meta)

    print(f"S1d: {n_mod:,} models, {n_res:,} residues, mean pLDDT {mean_res:.2f}, "
          f"{frac70*100:.2f}% >= {PLDDT_CUT:.0f}")
    print(f"S1e: {len(keep)} strains with >=5 models; grand mean {gmean:.2f}; "
          f"strain medians {keep.med.min():.1f}–{keep.med.max():.1f}")
    print(f"S1f: {len(d)} strains; genome {d.Mb.min():.2f}–{d.Mb.max():.2f} Mb; "
          f"proteome {int(d.protein_num.min())}–{int(d.protein_num.max())}; "
          f"{int(d.diazotroph.sum())} diazotrophs, {int(d.freshwater.sum())} freshwater")
    print("\nlowest-confidence strains:")
    print(keep.head(5).round(2).to_string())


if __name__ == "__main__":
    main()
