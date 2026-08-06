#!/usr/bin/env python3
"""Is the reaction-centre surface more conserved than its own overall CR already implies?

THE OBJECTION
-------------
Reaction-centre families have high conservation everywhere: complete-protein CR ~0.86, core CR
0.892, surface CR 0.834.  Because CR is bounded at 1.0, a family whose core sits at 0.892 has at
most 0.108 of room left, so its core-minus-surface gap is arithmetically forced to be small.  The
observed narrowing of the gradient towards the reaction centre (+0.210 -> +0.117 -> +0.037) might
therefore be a ceiling artefact rather than a statement about surfaces.

FOUR TESTS, from weakest assumption to strongest
------------------------------------------------
1.  CR-MATCHED CONTROLS.  For each reaction-centre family, take the non-photosynthetic families
    whose complete-protein CR is closest, and compare surface CR and the core-surface gap.  If
    matched controls show the same small gap, the effect is a ceiling artefact.
2.  LOGIT SCALE.  Repeat the three-class gap comparison after logit-transforming CR, which removes
    the compression of differences near the bound.  A ceiling artefact disappears on this scale;
    a real difference in relative position does not.
3.  REGRESSION ON CORE CR.  Model surface CR from core CR (allowing curvature) plus protein length,
    and ask whether reaction-centre identity still carries an independent positive coefficient.
    This asks exactly the reviewer's question: is the surface higher than the core predicts?
4.  RESIDUAL RANKING.  The residual of test 3 for every family, so the reaction-centre families can
    be located within the whole panel rather than only against a chosen control set.

Input  : 40.whole_CR-MR/panorama_core_surface_351.csv, Figures/Figure4/buried_fraction_351.csv
Output : printed report + rc_surface_ceiling_tests.tsv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

ROOT = Path("/home/yangyicheng/27.evolution")
HERE = Path(__file__).resolve().parent
RC = {"psbA", "psbC", "psbD", "psaA", "psaB", "psaC"}
PETC = {"Photosystem II", "Photosystem I", "Cytochrome b6f", "ATP synthase",
        "NADH dehydrogenase (NDH-1)"}
EPS = 1e-3


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load():
    d = pd.read_csv(ROOT / "40.whole_CR-MR/panorama_core_surface_351.csv").dropna(
        subset=["core_CR", "surface_CR", "complete_CR"])
    n = pd.read_csv(HERE / "buried_fraction_351.csv")[["unit", "n_res"]]
    d = d.merge(n, on="unit", how="left")
    d["cls"] = np.where(d.unit.isin(RC), "RC core",
                        np.where(d.pathway.isin(PETC), "other PETC", "background"))
    d["gap"] = d.core_CR - d.surface_CR
    d["gap_logit"] = logit(d.core_CR) - logit(d.surface_CR)
    return d


def t1_matched(d, k=8):
    print("TEST 1 — CR-matched controls\n" + "-" * 74)
    rc = d[d.cls == "RC core"]
    bg = d[d.cls == "background"].copy()
    print(f"{'RC family':10s} {'complete':>9s} {'surface':>8s} {'gap':>7s} │ "
          f"{'matched bg complete':>19s} {'surface':>8s} {'gap':>7s}")
    rows = []
    for _, r in rc.iterrows():
        m = bg.assign(dist=(bg.complete_CR - r.complete_CR).abs()).nsmallest(k, "dist")
        print(f"{r.unit:10s} {r.complete_CR:9.3f} {r.surface_CR:8.3f} {r.gap:7.3f} │ "
              f"{m.complete_CR.median():19.3f} {m.surface_CR.median():8.3f} "
              f"{m.gap.median():7.3f}")
        rows.append(dict(unit=r.unit, complete=r.complete_CR, rc_surface=r.surface_CR,
                         rc_gap=r.gap, ctrl_complete=m.complete_CR.median(),
                         ctrl_surface=m.surface_CR.median(), ctrl_gap=m.gap.median(),
                         n_ctrl=len(m)))
    t = pd.DataFrame(rows)
    ds, dg = t.rc_surface - t.ctrl_surface, t.rc_gap - t.ctrl_gap
    ps = wilcoxon(ds).pvalue if len(t) > 2 else np.nan
    pg = wilcoxon(dg).pvalue if len(t) > 2 else np.nan
    print(f"\n  surface CR, RC minus matched controls : median {ds.median():+.3f}  "
          f"({int((ds > 0).sum())}/{len(t)} higher)  Wilcoxon P = {ps:.3f}")
    print(f"  core-surface gap, RC minus controls   : median {dg.median():+.3f}  "
          f"({int((dg < 0).sum())}/{len(t)} narrower)  Wilcoxon P = {pg:.3f}")
    print("  reading: if the gap difference vanishes here, the narrowing is a ceiling artefact\n")
    return t


def t2_logit(d):
    print("TEST 2 — the same three-class comparison on the logit scale\n" + "-" * 74)
    order = ["RC core", "other PETC", "background"]
    print(f"{'class':12s} {'n':>4s} {'gap (raw)':>10s} {'gap (logit)':>12s}")
    for c in order:
        s = d[d.cls == c]
        print(f"{c:12s} {len(s):4d} {s.gap.median():10.3f} {s.gap_logit.median():12.3f}")
    for col, lab in (("gap", "raw"), ("gap_logit", "logit")):
        H, p = kruskal(*[d[d.cls == c][col] for c in order])
        pr = mannwhitneyu(d[d.cls == "RC core"][col],
                          d[d.cls == "background"][col]).pvalue
        pe = mannwhitneyu(d[d.cls == "other PETC"][col],
                          d[d.cls == "background"][col]).pvalue
        print(f"  {lab:5s}: Kruskal-Wallis P = {p:.2e} │ RC vs background P = {pr:.4f} │ "
              f"other PETC vs background P = {pe:.5f}")
    print("  reading: a ceiling artefact should shrink or vanish on the logit scale\n")


def t3_regression(d):
    print("TEST 3 — does reaction-centre identity predict surface CR beyond core CR?\n" + "-" * 74)
    q = d.dropna(subset=["n_res"]).copy()
    q["rc"] = (q.cls == "RC core").astype(float)
    q["petc"] = (q.cls == "other PETC").astype(float)
    q["logL"] = np.log10(q.n_res)
    specs = [("surface ~ core", ["core_CR"]),
             ("surface ~ core + core²", ["core_CR", "core2"]),
             ("surface ~ core + core² + logL", ["core_CR", "core2", "logL"]),
             ("surface ~ core + core² + logL + PETC", ["core_CR", "core2", "logL", "petc"])]
    q["core2"] = q.core_CR ** 2
    out = []
    for lab, cols in specs:
        X = np.column_stack([np.ones(len(q))] + [q[c].values for c in cols] + [q.rc.values])
        y = q.surface_CR.values
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        dof = len(q) - X.shape[1]
        s2 = (resid ** 2).sum() / dof
        se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
        b_rc, se_rc = beta[-1], se[-1]
        tstat = b_rc / se_rc
        from scipy.stats import t as tdist
        p = 2 * tdist.sf(abs(tstat), dof)
        r2 = 1 - (resid ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  {lab:38s} β(RC) = {b_rc:+.4f} ± {se_rc:.4f}   t = {tstat:+.2f}   "
              f"P = {p:.4f}   R² = {r2:.3f}")
        out.append(dict(model=lab, beta_RC=b_rc, se=se_rc, t=tstat, P=p, R2=r2, n=len(q)))
    print("  reading: a positive, significant β(RC) means the reaction-centre surface is more\n"
          "  conserved than its own core CR, size and module class already predict\n")
    return pd.DataFrame(out), q


def t4_residuals(q):
    print("TEST 4 — where the five RC families sit in the panel-wide residual\n" + "-" * 74)
    X = np.column_stack([np.ones(len(q)), q.core_CR, q.core_CR ** 2, np.log10(q.n_res)])
    beta, *_ = np.linalg.lstsq(X, q.surface_CR.values, rcond=None)
    q = q.assign(resid=q.surface_CR.values - X @ beta)
    q = q.assign(pct=q.resid.rank(pct=True) * 100)
    rc = q[q.cls == "RC core"].sort_values("resid", ascending=False)
    for _, r in rc.iterrows():
        print(f"  {r.unit:8s} surface {r.surface_CR:.3f}  predicted {r.surface_CR - r.resid:.3f}  "
              f"residual {r.resid:+.3f}  → {r.pct:.0f}th percentile of {len(q)} families")
    for c in ("RC core", "other PETC", "background"):
        s = q[q.cls == c]
        print(f"  {c:12s} median residual {s.resid.median():+.4f}  (n = {len(s)})")
    p = mannwhitneyu(q[q.cls == "RC core"].resid, q[q.cls == "background"].resid).pvalue
    p2 = mannwhitneyu(q[q.cls == "other PETC"].resid, q[q.cls == "background"].resid).pvalue
    print(f"  RC vs background residual P = {p:.4f} │ other PETC vs background P = {p2:.5f}\n")
    return q


def main():
    d = load()
    print(f"{len(d)} families with a core/surface partition "
          f"(RC {int((d.cls == 'RC core').sum())}, other PETC "
          f"{int((d.cls == 'other PETC').sum())}, background "
          f"{int((d.cls == 'background').sum())})\n")
    print("class medians of complete-protein CR: " + ", ".join(
        f"{c} {d[d.cls == c].complete_CR.median():.3f}"
        for c in ("RC core", "other PETC", "background")) + "\n")
    t1 = t1_matched(d)
    t2_logit(d)
    t3, q = t3_regression(d)
    q = t4_residuals(q)
    with open(HERE / "rc_surface_ceiling_tests.tsv", "w") as fh:
        fh.write("# TEST 1: CR-matched controls (8 nearest background families per RC family)\n")
        t1.to_csv(fh, sep="\t", index=False, float_format="%.4g")
        fh.write("\n# TEST 3: surface CR regression, coefficient on reaction-centre identity\n")
        t3.to_csv(fh, sep="\t", index=False, float_format="%.4g")
        fh.write("\n# TEST 4: per-family residuals from surface ~ core + core^2 + log length\n")
        q[["unit", "pathway", "cls", "complete_CR", "core_CR", "surface_CR", "gap",
           "gap_logit", "n_res", "resid", "pct"]].to_csv(
            fh, sep="\t", index=False, float_format="%.4g")
    print("wrote rc_surface_ceiling_tests.tsv")


if __name__ == "__main__":
    main()
