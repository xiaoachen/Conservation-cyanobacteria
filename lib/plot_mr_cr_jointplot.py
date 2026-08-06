#!/usr/bin/env python3
import os
import glob
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress


def sci_pvalue_str(p: float) -> str:
    """Format P-value in scientific style, e.g., '1 × 10^-41'."""
    if p <= 0:
        return "≈ 0"
    if p >= 0.001:
        return f"{p:.3f}"
    s = f"{p:.1e}"
    mantissa, exp = s.split("e")
    exp = int(exp)
    return f"{float(mantissa):.1f} × 10^{exp}"


def load_mr_cr_from_dir(d: str, pattern: str = "*_mrcr_results.csv") -> pd.DataFrame:
    """
    Read all CSVs in a directory and return a DataFrame of per-file
    mean MR and mean CR (one row per CSV), after filtering out
    orthogroup-median and invalid rows. 其他逻辑保持不变。
    """
    files = sorted(glob.glob(os.path.join(d, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSV files matching '{pattern}' in: {d}")

    rows = []
    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[WARN] Failed reading {fp}: {e}")
            continue

        if not {"MR", "CR"}.issubset(df.columns):
            print(f"[WARN] Skip {fp}: missing MR/CR columns.")
            continue

        # 过滤掉正交组中位数伪行以及缺失/无效值
        mask_median = (
            df.get("target", pd.Series([None] * len(df))).astype(str).eq("__ORTHOGROUP_MEDIAN__")
            | df.get("target_file", pd.Series([None] * len(df))).astype(str).str.contains("__ORTHOGROUP_MEDIAN__", na=False)
            | df.get("target", pd.Series([None] * len(df))).astype(str).eq("NA")
        )
        # 保留所有列以便计算长度均值
        df = df.loc[~mask_median, :]

        # 转数值以便长度过滤和均值计算
        df["MR"] = pd.to_numeric(df["MR"], errors="coerce")
        df["CR"] = pd.to_numeric(df["CR"], errors="coerce")
        if "ref_length" in df.columns:
            df["ref_length"] = pd.to_numeric(df["ref_length"], errors="coerce")
        if "mapped_length" in df.columns:
            df["mapped_length"] = pd.to_numeric(df["mapped_length"], errors="coerce")

        # 先按 MR/CR 合法范围与缺失过滤
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["MR", "CR"])  # 仅按MR/CR过滤缺失
        df = df[(df["MR"] >= 0) & (df["MR"] <= 1) & (df["CR"] >= 0) & (df["CR"] <= 1)]

        # 再按长度比例过滤：mapped_length >= 0.3 * ref_length
        if {"ref_length", "mapped_length"}.issubset(df.columns):
            len_mask = (df["ref_length"].notna() & df["mapped_length"].notna())
            df = df[len_mask]
            df = df[df["mapped_length"] >= 0.3 * df["ref_length"]]

        if len(df) == 0:
            print(f"[WARN] No valid MR/CR rows after filtering in: {fp}")
            continue

        mean_mr = float(df["MR"].mean())
        mean_cr = float(df["CR"].mean())

        # 其他平均长度信息（如有列）
        # 转为数值以防字符串列，NaN将被自动忽略
        mean_ref_len = (
            float(df["ref_length"].mean()) if "ref_length" in df.columns else np.nan
        )
        mean_map_len = (
            float(df["mapped_length"].mean()) if "mapped_length" in df.columns else np.nan
        )

        base = os.path.basename(fp)
        og_id = base.split("_mrcr_results.csv")[0]
        rows.append({
            "OG": og_id,
            "file": base,
            "MR": mean_mr,
            "CR": mean_cr,
            "mean_ref_length": mean_ref_len,
            "mean_mapped_length": mean_map_len,
            "count": int(len(df)),
        })

    if not rows:
        raise ValueError("No valid per-file MR/CR means found.")
    return pd.DataFrame(rows)


def plot_joint(df: pd.DataFrame, output: str, dpi: int = 300, title_panel: str = ""):
    """Joint scatter of MR vs CR with marginal densities and stats."""
    sns.set_theme(style="white", context="talk")

    rho, pval = spearmanr(df["CR"], df["MR"])
    lr = linregress(df["CR"], df["MR"])
    x_min, x_max = 0.0, 1.0
    x_line = np.linspace(x_min, x_max, 200)
    y_line = lr.intercept + lr.slope * x_line

    g = sns.JointGrid(data=df, x="CR", y="MR", xlim=(0, 1), ylim=(0, 1), height=6)
    g.plot_joint(sns.scatterplot, s=18, color="#3A7BCF", alpha=0.7, edgecolor="none")
    g.plot_marginals(sns.kdeplot, color="#3A7BCF", fill=True, alpha=0.35)

    g.ax_joint.plot([0, 1], [0, 1], linestyle=":", color="black", linewidth=1.2)
    g.ax_joint.plot(x_line, y_line, linestyle="-", color="black", linewidth=1.2)

    x_med = float(df["CR"].median())
    y_med = float(df["MR"].median())
    for ax in (g.ax_joint, g.ax_marg_x, g.ax_marg_y):
        if ax is g.ax_marg_x:
            ax.axvline(x_med, linestyle="--", color="black", alpha=0.7)
        elif ax is g.ax_marg_y:
            ax.axhline(y_med, linestyle="--", color="black", alpha=0.7)
        else:
            ax.axvline(x_med, linestyle="--", color="black", alpha=0.25)
            ax.axhline(y_med, linestyle="--", color="black", alpha=0.25)

    g.set_axis_labels("Conservation ratio (CR)", "Mapping ratio (MR)")
    g.ax_joint.set_xlim(0, 1)
    g.ax_joint.set_ylim(0, 1)

    text = f"Spearman's ρ = {rho:.2f}\nP < {sci_pvalue_str(pval)}"
    g.ax_joint.text(0.05, 0.10, text, transform=g.ax_joint.transAxes, fontsize=12, color="black")

    # Do not draw panel letter by default (title_panel is empty)
    if title_panel:
        g.fig.text(0.01, 0.99, title_panel, ha="left", va="top", fontsize=16)

    sns.despine(fig=g.fig)
    g.fig.tight_layout()

    # Save PNG and PDF side-by-side
    base, ext = os.path.splitext(output)
    pdf_path = base + ".pdf"
    g.fig.savefig(output, dpi=dpi)
    g.fig.savefig(pdf_path)
    print(f"[OK] Saved joint figure to: {output} and {pdf_path}")
    print(f"[INFO] N={len(df)}, Spearman rho={rho:.4f}, p={pval:.3e}, slope={lr.slope:.3f}, intercept={lr.intercept:.3f}")


def plot_distribution(df: pd.DataFrame, column: str, output: str, dpi: int = 300):
    """Single-variable density plot with median line for CR or MR."""
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df, x=column, fill=True, color="#3A7BCF", alpha=0.35, ax=ax)
    median_val = float(df[column].median())
    ax.axvline(median_val, linestyle="--", color="black", alpha=0.7)
    ax.set_xlim(0, 1)
    xlabel = "Conservation ratio (CR)" if column == "CR" else "Mapping ratio (MR)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title(f"{xlabel} distribution")
    sns.despine()
    fig.tight_layout()

    # Save PNG and PDF side-by-side
    base, ext = os.path.splitext(output)
    pdf_path = base + ".pdf"
    fig.savefig(output, dpi=dpi)
    fig.savefig(pdf_path)
    print(f"[OK] Saved {column} distribution to: {output} and {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MR and CR from CSVs.")
    parser.add_argument("--dir", default="/home/yangyicheng/27.evolution/02.alphafold3/04.single_mrcr_1112/Result",
                        help="Directory containing *_mrcr_results.csv")
    parser.add_argument("--pattern", default="*_mrcr_results.csv", help="Glob pattern for CSV files")
    parser.add_argument("--mode", choices=["separate", "joint", "both"], default="separate",
                        help="Plot mode: separate distributions, joint scatter, or both")
    parser.add_argument("--output", default="mr_cr_jointplot.png", help="Output image path for joint plot")
    parser.add_argument("--output-prefix", default="mr_cr", help="Prefix for separate plots (creates <prefix>_CR_distribution.png and <prefix>_MR_distribution.png)")
    parser.add_argument("--dpi", type=int, default=600, help="Output figure DPI")
    parser.add_argument("--summary-csv", default=None,
                        help="Path to save aggregated per-file MR/CR and length summaries (default: <dir>/<output-prefix>_aggregated.csv)")
    args = parser.parse_args()

    df = load_mr_cr_from_dir(args.dir, args.pattern)

    # Resolve output paths
    joint_output = args.output
    if not os.path.isabs(joint_output):
        joint_output = os.path.join(args.dir, joint_output)

    prefix_path = args.output_prefix
    if not os.path.isabs(prefix_path):
        prefix_path = os.path.join(args.dir, prefix_path)

    # 保存聚合结果 CSV（每个文件一行，含均值与附加信息）
    summary_csv = args.summary_csv
    if not summary_csv:
        summary_csv = os.path.join(args.dir, f"{args.output_prefix}_aggregated.csv")
    elif not os.path.isabs(summary_csv):
        summary_csv = os.path.join(args.dir, summary_csv)

    try:
        df.to_csv(summary_csv, index=False)
        print(f"[OK] Saved aggregated summary CSV: {summary_csv}")
    except Exception as e:
        print(f"[WARN] Failed saving summary CSV: {e}")

    # Dispatch by mode
    if args.mode in ("separate", "both"):
        cr_out = f"{prefix_path}_CR_distribution.png"
        mr_out = f"{prefix_path}_MR_distribution.png"
        plot_distribution(df, "CR", cr_out, dpi=args.dpi)
        plot_distribution(df, "MR", mr_out, dpi=args.dpi)

    if args.mode in ("joint", "both"):
        plot_joint(df, joint_output, dpi=args.dpi)


if __name__ == "__main__":
    main()