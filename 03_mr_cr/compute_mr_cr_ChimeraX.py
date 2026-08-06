#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, glob, argparse, csv, re, subprocess, tempfile, shutil, uuid, time
import numpy as np
import gemmi

# ---- utils ----

def extract_protein_name(path):
    m = re.search(r'(WP_\d+(?:\.\d+)?)', path, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    parent = os.path.basename(os.path.dirname(path))
    m2 = re.search(r'(WP_\d+(?:\.\d+)?)', parent, re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    parent2 = os.path.basename(os.path.dirname(os.path.dirname(path)))
    m3 = re.search(r'(WP_\d+(?:\.\d+)?)', parent2, re.IGNORECASE)
    if m3:
        return m3.group(1).upper()
    return os.path.splitext(os.path.basename(path))[0]

def discover_cifs(root):
    patt = os.path.join(root, "**", "model.cif")
    paths = sorted(glob.glob(patt, recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(root, "**", "*.cif"), recursive=True))
    return [p for p in paths if os.path.isfile(p)]

def read_cif_structure(cif_path, model_index=0, chain_id=None):
    """
    读取 mmCIF 模型，抽取序列与按序的 Cα 坐标。
    返回:
      seq: str (one-letter)
      ca_xyz: (N,3) float ndarray
      res_ids: list of (seq_num, icode)
      plddt: (N,) float ndarray (pLDDT 置信度)
    """
    st = gemmi.read_structure(cif_path)
    if model_index >= len(st):
        raise ValueError(f"{cif_path}: model_index {model_index} out of range. Models: {len(st)}")
    model = st[model_index]

    # 选链：优先指定链；否则取第一个含有CA的聚合肽链
    chosen_chain = None
    if chain_id is not None:
        for ch in model:
            if ch.name.strip() == str(chain_id):
                chosen_chain = ch
                break
        if chosen_chain is None:
            raise ValueError(f"{cif_path}: chain {chain_id} not found.")
    else:
        for ch in model:
            # 仅选择蛋白链
            if any(res.get_ca() is not None for res in ch):
                chosen_chain = ch
                break
    if chosen_chain is None:
        raise ValueError(f"{cif_path}: no protein chain with CA atoms found.")

    seq = []
    ca_xyz = []
    plddt = []
    res_ids = []
    for res in chosen_chain:
        ca = res.get_ca()
        if ca is None:
            continue
        # 转 one-letter（非标准残基转 X 并跳过）
        try:
            one = gemmi.find_tabulated_residue(res.name).one_letter_code
        except Exception:
            one = 'X'
        if one == 'X':
            # 跳过非常见残基
            continue
        seq.append(one)
        p = ca.pos
        ca_xyz.append([p.x, p.y, p.z])
        
        # 获取pLDDT值（AlphaFold将置信度存储在b_iso字段）
        try:
            b_factor = ca.b_iso
        except AttributeError:
            # 兼容不同CIF格式
            b_factor = ca.b
        plddt_value = float(b_factor)
        plddt.append(plddt_value)
        
        res_ids.append((res.seqid.num, res.seqid.icode))

    return ''.join(seq), np.asarray(ca_xyz, dtype=float), res_ids, np.asarray(plddt, dtype=float)

def map_by_geometry_unique(ref_ca, trg_ca, cutoff=2.0):
    """
    一对一几何映射函数，确保每个参考位点最多对应一个目标位点
    参数:
      ref_ca: 参考结构的Cα坐标 (N_ref x 3)
      trg_ca: 目标结构的Cα坐标 (N_trg x 3)
      cutoff: Cα距离阈值(Å)
    返回:
      pairs: list of (i_ref, j_trg) 一对一映射对
    """
    n_ref, n_trg = len(ref_ca), len(trg_ca)
    # 计算所有Cα对之间的平方距离
    d2 = ((ref_ca[:, None, :] - trg_ca[None, :, :]) ** 2).sum(axis=2)
    
    used_trg = set()  # 记录已被映射的目标位点
    pairs = []        # 存储映射对
    
    # 按参考位点的最小距离排序（距离小的优先分配）
    order = np.argsort(d2.min(axis=1))
    
    for i in order:
        # 找到当前参考位点最近的目标位点
        j = int(d2[i].argmin())
        # 检查距离是否在阈值内且目标位点未被使用
        if d2[i, j] < (cutoff ** 2) and j not in used_trg:
            pairs.append((int(i), int(j)))
            used_trg.add(j)  # 标记该目标位点已被使用
    
    return pairs

def call_chimerax_matchmaker(ref_path, target_path, output_path, cutoff=2.0, debug=False):
    """
    调用 ChimeraX 的 matchmaker 函数进行结构比对，使用正确的命令格式
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    base_dir = os.path.dirname(output_path)
    base_name = os.path.basename(output_path)
    root, ext = os.path.splitext(base_name)
    suffix = f"{int(time.time()*1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    final_path = os.path.join(base_dir, f"{root}.{suffix}{ext}")
    cmd = [
        'ChimeraX',
        '--nogui',
        '--cmd', f'open "{ref_path}"',
        '--cmd', f'open "{target_path}"',
        '--cmd', f'matchmaker #2 to #1',
        '--cmd', f'save "{final_path}" format mmcif models #2',
        '--cmd', 'close all',
        '--cmd', 'exit'
    ]
    
    try:
        if debug:
            print(f"[DEBUG] Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if not os.path.exists(final_path):
            print(f"[WARN] Output file {final_path} was not created, but ChimeraX returned success")
            return None
        if debug:
            print(f"[SUCCESS] Aligned structure saved to {final_path}")
        return final_path
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ChimeraX failed with return code {e.returncode}")
        print(f"[ERROR] ChimeraX stdout: {e.stdout}")
        print(f"[ERROR] ChimeraX stderr: {e.stderr}")
        return None
    except FileNotFoundError:
        print("[ERROR] ChimeraX not found in PATH. Please ensure ChimeraX is installed and added to PATH.")
        return None

def compute_mr_cr_aligned(ref_seq, ref_ca, ref_plddt, aligned_seq, aligned_ca, aligned_plddt, cutoff=2.0, plddt_min=70.0):
    """
    计算对齐结构的 MR 和 CR，使用一对一几何映射，加入pLDDT掩蔽
    参数:
      ref_seq, ref_ca, ref_plddt: 参考结构的序列、Cα坐标和pLDDT值
      aligned_seq, aligned_ca, aligned_plddt: 对齐后的目标结构的序列、Cα坐标和pLDDT值
      cutoff: Cα距离阈值(Å)
      plddt_min: pLDDT过滤阈值
    返回:
      MR, CR, mapped_positions (boolean array), status
    """
    # 1. pLDDT掩蔽：筛选高置信度残基
    ref_mask = ref_plddt >= plddt_min
    aligned_mask = aligned_plddt >= plddt_min
    
    # 创建映射索引
    ref_idx = np.where(ref_mask)[0]
    aligned_idx = np.where(aligned_mask)[0]
    
    if len(ref_idx) == 0 or len(aligned_idx) == 0:
        return 0.0, 0.0, np.zeros(len(ref_seq), dtype=bool), f"no_high_plddt_residues (plddt_min={plddt_min})"
    
    # 2. 子集化：仅使用高置信度残基
    ref_ca_filtered = ref_ca[ref_idx]
    ref_seq_filtered = ''.join([ref_seq[i] for i in ref_idx])
    
    aligned_ca_filtered = aligned_ca[aligned_idx]
    aligned_seq_filtered = ''.join([aligned_seq[j] for j in aligned_idx])
    
    # 3. 一对一几何映射（仅在高置信度残基间）
    pairs = map_by_geometry_unique(ref_ca_filtered, aligned_ca_filtered, cutoff=cutoff)
    
    # 4. 初始化映射数组（基于原始参考结构）
    mapped = np.zeros(len(ref_seq), dtype=bool)
    conserved = np.zeros(len(ref_seq), dtype=bool)
    
    # 5. 填充映射和保守性信息
    for i_filt, j_filt in pairs:
        i = ref_idx[i_filt]  # 转回原始索引
        j = aligned_idx[j_filt]
        
        if i < len(ref_seq) and j < len(aligned_seq):
            mapped[i] = True
            if ref_seq[i] == aligned_seq[j]:
                conserved[i] = True
    
    # 6. 计算MR和CR（分母使用高置信度残基数）
    high_conf_ref_count = len(ref_idx)
    mapped_count = mapped.sum()
    
    # MR = 映射的高置信度残基数 / 总高置信度残基数
    MR = mapped_count / high_conf_ref_count if high_conf_ref_count > 0 else 0.0
    
    # CR = 保守的映射残基数 / 总映射残基数
    conserved_count = conserved.sum()
    CR = conserved_count / mapped_count if mapped_count > 0 else 0.0
    
    status = "ok" if mapped_count > 0 else "no_mapping"
    status += f"; pLDDT≥{plddt_min}: ref={high_conf_ref_count}/{len(ref_seq)}, target={len(aligned_idx)}/{len(aligned_seq)}"
    
    return MR, CR, mapped, status

def main():
    ap = argparse.ArgumentParser(description="Compute MR and CR using ChimeraX matchmaker for structure alignment with one-to-one geometric mapping and pLDDT masking.")
    ap.add_argument("--ref", default=None, help="参考结构的 mmCIF 文件路径（当不提供且给定 --root 时自动选择）")
    ap.add_argument("--targets", nargs="+", default=None,
                    help="待比较的其它物种 mmCIF（一个或多个）。可用通配符；若未提供且给定 --root，则自动使用 root 下的所有 mmCIF")
    ap.add_argument("--root", default=None, help="测试数据根目录（例如 01.test），用于自动发现 mmCIF")
    ap.add_argument("--exclude-self", action="store_true", help="从 targets 排除参考文件自身")
    ap.add_argument("--ref-chain", default=None, help="参考链 ID（默认自动选首个含 CA 的蛋白链）")
    ap.add_argument("--trg-chain", default=None, help="目标链 ID（默认自动选首个含 CA 的蛋白链）")
    ap.add_argument("--model-index", type=int, default=0, help="选用的模型（0 表示 rank1 模型）")
    ap.add_argument("--cutoff", type=float, default=2.0, help="Cα 映射阈值（Å），论文使用 2 Å")
    ap.add_argument("--plddt-min", type=float, default=70.0, help="pLDDT过滤阈值（默认70.0），低于此值的残基将被排除在分析之外")
    ap.add_argument("--out", default="mr_cr_results.csv", help="输出 CSV 文件")
    ap.add_argument("--temp-dir", default=None, help="临时目录用于存储对齐后的结构（默认在当前目录创建临时目录）")
    ap.add_argument("--debug", action="store_true", default=False, help="启用调试输出")
    args = ap.parse_args()

    stale = [d for d in os.listdir('.') if d.startswith('chimera_align_') and os.path.isdir(d)]
    for d in stale:
        try:
            shutil.rmtree(d)
            if args.debug:
                print(f"[DEBUG] Removed stale temp directory: {d}")
        except Exception as e:
            print(f"[ERROR] Failed to remove stale temp directory {d}: {e}")
    # 基于 --root/显式参数确定参考与目标
    if args.root:
        all_cifs = discover_cifs(args.root)
        if not all_cifs:
            raise RuntimeError(f"No mmCIF found under root: {args.root}")
        if args.ref is None:
            seed0 = [p for p in all_cifs if "seed-1_sample-0" in p]
            args.ref = seed0[0] if seed0 else all_cifs[0]
        if args.targets is None:
            args.targets = all_cifs
    else:
        if args.ref is None or args.targets is None:
            raise SystemExit("必须提供 --root，或同时提供 --ref 与 --targets")

    target_paths = sorted(sum([glob.glob(p) for p in args.targets], []))
    if args.exclude_self and args.ref is not None:
        ref_real = os.path.realpath(args.ref)
        target_paths = [p for p in target_paths if os.path.realpath(p) != ref_real]

    if not target_paths:
        raise RuntimeError("所选 targets 为空，请检查 --targets 或 --root")

    print(f"[INFO] Reference: {args.ref}")
    print(f"[INFO] Targets: {len(target_paths)} files")
    print(f"[INFO] pLDDT threshold: {args.plddt_min}")

    # 读取参考结构
    ref_seq, ref_ca, _, ref_plddt = read_cif_structure(args.ref, model_index=args.model_index, chain_id=args.ref_chain)
    if len(ref_seq) == 0 or ref_ca.shape[0] == 0:
        raise RuntimeError(f"Reference empty: {args.ref}")

    # 获取参考结构名称
    ref_name = extract_protein_name(args.ref)

    # 创建临时目录用于存储对齐结构（放在当前目录）
    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        temp_dir = tempfile.mkdtemp(dir='.', prefix='chimera_align_')
    os.makedirs(temp_dir, exist_ok=True)
    if args.debug:
        print(f"[INFO] Using temp directory: {temp_dir}")

    rows = []
    # 修改表头，增加pLDDT相关信息
    rows.append(["ref", "target", "target_file", "MR", "CR", "ref_length", "high_plddt_ref", "mapped_length", "status"])
    mrs, crs = [], []

    for tf in target_paths:
        tgt_name = extract_protein_name(tf)
        if args.debug:
            print(f"[PROCESSING] {tgt_name}")
        
        try:
            # 1. 用 ChimeraX matchmaker 进行结构比对
            aligned_path = os.path.join(temp_dir, f"aligned_{tgt_name}.cif")
            final_aligned = call_chimerax_matchmaker(args.ref, tf, aligned_path, cutoff=args.cutoff, debug=args.debug)
            if not final_aligned or not os.path.exists(final_aligned):
                raise RuntimeError(f"ChimeraX matchmaker failed for {tf}")
            
            # 2. 读取对齐后的结构
            aligned_seq, aligned_ca, _, aligned_plddt = read_cif_structure(final_aligned, model_index=args.model_index, chain_id=args.trg_chain)
            
            # 3. 计算 MR 和 CR
            MR, CR, mapped, status = compute_mr_cr_aligned(
                ref_seq, ref_ca, ref_plddt,
                aligned_seq, aligned_ca, aligned_plddt,
                cutoff=args.cutoff,
                plddt_min=args.plddt_min
            )
            
            # 记录高置信度残基数
            high_plddt_ref = np.sum(ref_plddt >= args.plddt_min)
            
            rows.append([ref_name, tgt_name, os.path.basename(tf), f"{MR:.4f}", f"{CR:.4f}", 
                        str(len(ref_seq)), str(high_plddt_ref), str(mapped.sum()), status])
            mrs.append(MR)
            crs.append(CR)
            
        except Exception as e:
            high_plddt_ref = np.sum(ref_plddt >= args.plddt_min) if 'ref_plddt' in locals() else "NA"
            rows.append([ref_name, tgt_name, os.path.basename(tf), "NA", "NA", 
                        str(len(ref_seq)), str(high_plddt_ref), "0", f"error: {str(e)}"])
            sys.stderr.write(f"[ERROR] {tf}: {e}\n")

    # 计算正交群中位数
    og_mr = np.median(mrs) if len(mrs) else 0.0
    og_cr = np.median(crs) if len(crs) else 0.0
    rows.append([ref_name, "NA", "__ORTHOGROUP_MEDIAN__", f"{og_mr:.4f}", f"{og_cr:.4f}", 
                str(len(ref_seq)), str(high_plddt_ref), "NA", "NA"])

    # 保存结果
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    # 清理临时目录
    if args.temp_dir is None:
        shutil.rmtree(temp_dir)
        if args.debug:
            print(f"[INFO] Cleaned up temp directory: {temp_dir}")

    print(f"[OK] Wrote: {args.out}")
    print(f"Orthogroup median MR={og_mr:.4f}, CR={og_cr:.4f} (N={len(mrs)} targets)")

if __name__ == "__main__":
    main()