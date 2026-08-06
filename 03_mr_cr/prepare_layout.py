#!/usr/bin/env python3
"""prepare_layout.py v2 — 两阶段策略：
  Phase A: 继承 12.mrcr 02–11 模块的 gene-named 结构
           - 旧 GCF_000011545 WP 文件夹 -> 丢弃
           - 添加 GCF_000011345 替代 WP（通过 annotation 匹配从新 AF3 找）
           - 其余 WP 文件夹直接 symlink 到 12.mrcr 原 AF3
  Phase B: 用新 222 OG 重建 01.singlecopy_gene
           - 从 single_copy_core_OGs_222.csv 拿每个新 OG 的 36 WPs
           - AF3 来源优先 26.orthofinder/01.alphafold3，否则 12.mrcr (按 WP 全局查)
  Phase C: 三级 ref fallback (GCF_000009725 → GCF_000011345 → 最长成员)

输出: alt_ref_map.tsv, og_to_module_map.tsv, unmodeled.tsv, swap_log.tsv, map (每模块), prepare_layout.log
"""
import csv
import os
import re
import sys
import gzip
from pathlib import Path
from collections import defaultdict

# ---------- 路径 ----------
ROOT     = Path("/home/yangyicheng/27.evolution")
OLD_MRCR = ROOT / "12.mrcr-20251125"
NEW_MRCR = ROOT / "28.mrcr-20250602"
OFINDER  = ROOT / "26.orthofinder"
AF3_NEW_DIRS = [
    OFINDER / "01.alphafold3/test9/output",
    OFINDER / "01.alphafold3/test9.1/output",
]
PCC6803, GLOEO, OLD_REF = "GCF_000009725", "GCF_000011345", "GCF_000011545"

# ---------- 步骤 1: 索引所有 AF3 输出 (新 + 旧) ----------
print("[1/7] 索引 AF3 输出 (新 + 旧)...", file=sys.stderr)
af3_new = {}  # WP_x_GCF_y -> Path (top-level WP folder in new AF3)
for af3_root in AF3_NEW_DIRS:
    if not af3_root.is_dir():
        continue
    for batch in sorted(af3_root.glob("output_*")):
        for wp_dir in batch.iterdir():
            if wp_dir.is_dir() and wp_dir.name.startswith("WP_"):
                cif = wp_dir / wp_dir.name.lower() / "seed-1_sample-0" / "model.cif"
                if cif.is_file():
                    af3_new.setdefault(wp_dir.name, wp_dir)

# 旧 AF3 索引（遍历 12.mrcr 全部模块/gene/WP）
af3_old = {}  # WP_x_GCF_y -> Path (origin in 12.mrcr, e.g. 12.mrcr/02.PSII/psbA/WP_xxx_GCF_yyy)
for mod_dir in sorted(OLD_MRCR.iterdir()):
    if not mod_dir.is_dir() or not re.match(r"\d{2}\.", mod_dir.name):
        continue
    for sub in mod_dir.iterdir():
        if not sub.is_dir() or sub.name in ("Result", "Result_Structure"):
            continue
        # sub = gene 文件夹 (02-11) 或 OG 文件夹 (01.singlecopy)
        for wp_dir in sub.iterdir():
            if wp_dir.is_dir() and wp_dir.name.startswith("WP_"):
                cif = wp_dir / wp_dir.name.lower() / "seed-1_sample-0" / "model.cif"
                if cif.is_file():
                    af3_old.setdefault(wp_dir.name, wp_dir)

print(f"   新 AF3: {len(af3_new)} WP", file=sys.stderr)
print(f"   旧 AF3: {len(af3_old)} WP", file=sys.stderr)
print(f"   重叠 (同 WP 同时存在): {len(set(af3_new) & set(af3_old))}", file=sys.stderr)

def find_af3(wp):
    """返回 (path, source) 或 (None, None)。new 优先。"""
    if wp in af3_new:
        return af3_new[wp], "new"
    if wp in af3_old:
        return af3_old[wp], "old"
    return None, None

# ---------- 步骤 2: 读 OG 表 & 11345 替代映射 ----------
print("[2/7] 读 OG 表 & 11345 映射...", file=sys.stderr)

# new_OG -> [WPs in 222 OG set]
og_to_wps = defaultdict(list)
wp_to_og  = {}
og_anno   = {}
og_only_new = set()
for path, mark in [
    (OFINDER / "single_copy_core_OGs_222.csv", False),
    (OFINDER / "single_copy_core_OGs_only_new_36.csv", True),
]:
    with open(path) as fh:
        rd = csv.DictReader(fh)
        for row in rd:
            og, wp, anno = row["OG_cluster"].strip(), row["WP_accession"].strip(), row.get("annotation","").strip()
            wp_to_og[wp] = og
            og_to_wps[og].append(wp)
            og_anno.setdefault(og, anno)
            if mark:
                og_only_new.add(og)
print(f"   总 OG: {len(og_to_wps)}, only_new: {len(og_only_new)}, total WP: {len(wp_to_og)}", file=sys.stderr)

# GCF_000011345 → annotation 索引（来自 GCF_000011345_matched_all_final.csv）
gcf11345_proteins = []   # list of (WP, annotation, OG?)
with open(OFINDER / "GCF_000011345_matched_all_final.csv") as fh:
    rd = csv.DictReader(fh)
    for row in rd:
        wp = row.get("WP_accession", "").strip()
        anno = row.get("annotation", "").strip()
        og = row.get("OG_cluster", "").strip()
        gcf11345_proteins.append((wp, anno, og))
print(f"   GCF_000011345 候选蛋白: {len(gcf11345_proteins)}", file=sys.stderr)

# 同时从 af3_new 索引里把所有 11345 蛋白也加进来（防止有些 11345 蛋白没在 matched_all_final 中但已建模）
gcf11345_in_af3 = [wp for wp in af3_new if wp.endswith(f"_{GLOEO}")]
# 对于 af3 里的 11345，annotation 暂时未知，先标空
known_anno = {wp: anno for wp, anno, _ in gcf11345_proteins}
for wp in gcf11345_in_af3:
    if wp not in known_anno:
        gcf11345_proteins.append((wp, "", ""))
print(f"   GCF_000011345 in AF3: {len(gcf11345_in_af3)}", file=sys.stderr)

# ---------- 步骤 3: Phase A — 继承 12.mrcr 02–11 模块 ----------
print("[3/7] Phase A — 继承 02-11 模块 + 11545→11345 替换...", file=sys.stderr)

# 简单 annotation 归一化
def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", s.lower())).strip()

# 用 12.mrcr 02-11 模块的 GCF_000009725 WP 抓 gene→annotation
def get_gene_annotation(mod, gene):
    """从 12.mrcr/<mod>/<gene>/WP_xxx_GCF_000009725/wp_*/wp_*_data.json 或 None 拿不到时返回空"""
    gdir = OLD_MRCR / mod / gene
    # 找含 GCF_000009725 的 WP folder
    candidate_wp = None
    for wp_folder in gdir.iterdir():
        if wp_folder.is_dir() and PCC6803 in wp_folder.name:
            candidate_wp = wp_folder.name
            break
    if candidate_wp is None:
        # 退一步，找 OrthoFinder 库里包含该 WP 的注释
        for wp_folder in gdir.iterdir():
            if wp_folder.is_dir() and wp_folder.name.startswith("WP_"):
                if wp_folder.name in wp_to_og:
                    return og_anno.get(wp_to_og[wp_folder.name], ""), wp_folder.name
        return "", None
    # 直接查 wp_to_og + og_anno
    if candidate_wp in wp_to_og:
        return og_anno.get(wp_to_og[candidate_wp], ""), candidate_wp
    # 否则尝试从蛋白组 FASTA 抓，但太重；直接返回空字符串
    return "", candidate_wp

swap_rows = []     # mod, gene, old_wp(11545), new_wp(11345), match_method, annotation
unmodeled_rows = []
per_module_targets = defaultdict(list)

NAMED_MODULES = [
    "02.Photosystem_II", "03.Photosystem_I", "04.CBB_cycle",
    "05.Carbon_concentrating_mechanism", "06.ATP_synthase",
    "07.Cytochrome_b6f", "08.Photoprotection", "09.NDH",
    "10.Phycobilisome", "11.Pigment_biosynthesis",
]

for mod in NAMED_MODULES:
    mod_old = OLD_MRCR / mod
    mod_new = NEW_MRCR / mod
    if not mod_old.is_dir():
        continue
    genes = [p.name for p in mod_old.iterdir()
             if p.is_dir() and p.name not in ("Result", "Result_Structure")]
    for gene in sorted(genes):
        gdir_old = mod_old / gene
        gdir_new = mod_new / gene
        gdir_new.mkdir(parents=True, exist_ok=True)
        per_module_targets[mod].append(gene)

        # 复制/symlink 全部 WP folder（除 GCF_000011545）
        had_11545 = False
        for wp_folder in gdir_old.iterdir():
            if not (wp_folder.is_dir() and wp_folder.name.startswith("WP_")):
                continue
            if OLD_REF in wp_folder.name:
                had_11545 = True
                continue   # 丢弃 11545
            link = gdir_new / wp_folder.name
            if not link.exists():
                link.symlink_to(wp_folder.resolve())

        # 找 11345 替代
        gene_anno, ref_wp_old = get_gene_annotation(mod, gene)
        norm_anno = norm(gene_anno)
        replacement_wp = None
        method = None

        # 策略 A: 通过 OG 桥接 — 旧 OG 反查
        # 不容易做（12.mrcr 没显式 OG 列），跳过
        # 策略 B: 用 annotation 关键词在 GCF_11345 蛋白中匹配
        if gene_anno:
            for wp11345, a11345, _og in gcf11345_proteins:
                if not wp11345.endswith(f"_{GLOEO}"):
                    continue
                if a11345 and norm(a11345) == norm_anno:
                    replacement_wp = wp11345
                    method = "exact_anno"
                    break
            if replacement_wp is None:
                # 模糊：包含关键词
                keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", gene_anno)
                            if w.lower() not in ("protein", "subunit", "domain", "family", "chain", "type", "complex")][:3]
                if keywords:
                    for wp11345, a11345, _og in gcf11345_proteins:
                        if not wp11345.endswith(f"_{GLOEO}"):
                            continue
                        if all(re.search(rf"\b{re.escape(k)}\b", a11345, re.I) for k in keywords):
                            replacement_wp = wp11345
                            method = f"keyword:{'+'.join(keywords)}"
                            break

        # 策略 C: 用 gene 名直接搜
        if replacement_wp is None and gene:
            gpat = re.compile(rf"\b{re.escape(gene)}\b", re.I)
            for wp11345, a11345, _og in gcf11345_proteins:
                if not wp11345.endswith(f"_{GLOEO}"):
                    continue
                if a11345 and gpat.search(a11345):
                    replacement_wp = wp11345
                    method = "gene_name"
                    break

        if replacement_wp:
            src, _src_kind = find_af3(replacement_wp)
            if src:
                link = gdir_new / replacement_wp
                if not link.exists():
                    link.symlink_to(src.resolve())
                swap_rows.append((mod, gene, "11545→11345", replacement_wp, method, gene_anno))
            else:
                swap_rows.append((mod, gene, "11345_no_AF3", replacement_wp, method, gene_anno))
        elif had_11545:
            swap_rows.append((mod, gene, "no_replacement_found", "", "", gene_anno))

# ---------- 步骤 4: Phase B — 用 222 新 OG 重建 01.singlecopy_gene ----------
print("[4/7] Phase B — 重建 01.singlecopy_gene (222 新 OG)...", file=sys.stderr)
sc_module = "01.singlecopy_gene"
sc_dir = NEW_MRCR / sc_module
for og, wps in og_to_wps.items():
    og_dir = sc_dir / og
    og_dir.mkdir(parents=True, exist_ok=True)
    per_module_targets[sc_module].append(og)
    for wp in wps:
        if OLD_REF in wp:
            continue   # 不放 11545
        src, _src_kind = find_af3(wp)
        if src:
            link = og_dir / wp
            if not link.exists():
                link.symlink_to(src.resolve())
        else:
            unmodeled_rows.append((sc_module, og, wp))

# ---------- 步骤 5: 三级 ref 选择 ----------
print("[5/7] 三级 ref 选择...", file=sys.stderr)
alt_ref_rows = []

def get_length_from_cif(cif_path):
    if not cif_path.is_file():
        return 0
    n_ca = 0
    try:
        with open(cif_path) as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    fields = line.split()
                    if len(fields) > 3 and fields[3] == "CA":
                        n_ca += 1
    except Exception:
        return 0
    return n_ca

for mod, targets in per_module_targets.items():
    for target in sorted(set(targets)):
        tdir = NEW_MRCR / mod / target
        if not tdir.is_dir():
            continue
        wps = [p.name for p in tdir.iterdir() if p.is_symlink() and p.name.startswith("WP_")]
        ref_wp, tier = None, None
        # tier 1
        for wp in wps:
            if wp.endswith(f"_{PCC6803}"):
                ref_wp, tier = wp, 1; break
        # tier 2
        if ref_wp is None:
            for wp in wps:
                if wp.endswith(f"_{GLOEO}"):
                    ref_wp, tier = wp, 2; break
        # tier 3
        if ref_wp is None and wps:
            longest = max(
                wps,
                key=lambda w: get_length_from_cif(tdir / w / w.lower() / "seed-1_sample-0" / "model.cif")
            )
            ref_wp, tier = longest, 3
        if ref_wp:
            ref_path = str(tdir / ref_wp / ref_wp.lower() / "seed-1_sample-0" / "model.cif")
            alt_ref_rows.append((mod, target, tier, ref_wp, ref_path))

# ---------- 步骤 6: 写输出文件 ----------
print("[6/7] 写输出文件...", file=sys.stderr)
with open(NEW_MRCR / "alt_ref_map.tsv", "w") as fh:
    fh.write("module\ttarget\ttier\tref_wp\tref_path\n")
    for r in alt_ref_rows:
        fh.write("\t".join(str(x) for x in r) + "\n")

with open(NEW_MRCR / "swap_log.tsv", "w") as fh:
    fh.write("module\tgene\taction\treplacement_wp\tmethod\tannotation\n")
    for r in swap_rows:
        fh.write("\t".join(str(x) for x in r) + "\n")

with open(NEW_MRCR / "unmodeled.tsv", "w") as fh:
    fh.write("module\ttarget\tmissing_WP\n")
    for r in unmodeled_rows:
        fh.write("\t".join(r) + "\n")

with open(NEW_MRCR / "og_to_module_map.tsv", "w") as fh:
    fh.write("new_OG\tmodule\tcategory\tn_wps\tn_AF3\thas_PCC6803\thas_Gloeobacter\n")
    for og, wps in og_to_wps.items():
        n_af3 = sum(1 for wp in wps if (wp in af3_new or wp in af3_old))
        has_pcc = any(wp.endswith(f"_{PCC6803}") and (wp in af3_new or wp in af3_old) for wp in wps)
        has_gl  = any(wp.endswith(f"_{GLOEO}")  and (wp in af3_new or wp in af3_old) for wp in wps)
        cat = "only_new" if og in og_only_new else "common"
        fh.write(f"{og}\t{sc_module}\t{cat}\t{len(wps)}\t{n_af3}\t{has_pcc}\t{has_gl}\n")

# 每模块 map 文件
for mod, targets in per_module_targets.items():
    with open(NEW_MRCR / mod / "map", "w") as fh:
        for t in sorted(set(targets)):
            fh.write(t + "\n")

# ---------- 步骤 7: 摘要 ----------
print("[7/7] Summary", file=sys.stderr)
print(f"   总 target 数: {sum(len(v) for v in per_module_targets.values())}", file=sys.stderr)
for m in ["01.singlecopy_gene"] + NAMED_MODULES:
    targets = per_module_targets.get(m, [])
    n_links = 0
    if (NEW_MRCR / m).is_dir():
        for tdir in (NEW_MRCR / m).iterdir():
            if tdir.is_dir():
                n_links += sum(1 for x in tdir.iterdir() if x.is_symlink() and x.name.startswith("WP_"))
    print(f"   {m}: targets={len(set(targets))}, AF3_links={n_links}", file=sys.stderr)

print(f"\n   swap_log entries: {len(swap_rows)}", file=sys.stderr)
print(f"   unmodeled entries: {len(unmodeled_rows)}", file=sys.stderr)
print(f"   alt_ref entries: {len(alt_ref_rows)}", file=sys.stderr)
tier_dist = defaultdict(int)
for r in alt_ref_rows:
    tier_dist[r[2]] += 1
print(f"   ref tier 分布: {dict(tier_dist)}", file=sys.stderr)
