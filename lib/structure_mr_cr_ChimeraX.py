import os, sys, glob, argparse, csv, re, subprocess, tempfile, shutil, uuid, time, json
import numpy as np
import gemmi

# ---- utils ----

# MaxSASA (Empirical, e.g. Tien et al 2013 "Theoretical")
MAX_SASA = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0, 'E': 214.0, 'Q': 214.0, 'G': 97.0,
    'H': 216.0, 'I': 195.0, 'L': 191.0, 'K': 230.0, 'M': 214.0, 'F': 228.0, 'P': 154.0, 'S': 143.0,
    'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0, 'X': 1.0
}

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
      plddt: (N,) float ndarray (pLDDT 置信度 或 B-factor/SASA)
    """
    try:
        st = gemmi.read_structure(cif_path)
    except Exception as e:
        raise ValueError(f"Failed to read structure {cif_path}: {e}")

    if model_index >= len(st):
        # Fallback to model 0 if index is out of range (common in some files)
        model_index = 0
        
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
    res_ids = []
    plddt = []
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
        res_ids.append((res.seqid.num, res.seqid.icode))
        
        # 获取 B-factor (可能存储了 pLDDT 或 SASA)
        try:
            b_val = ca.b_iso
        except AttributeError:
            b_val = ca.b
        plddt.append(b_val)

    return ''.join(seq), np.asarray(ca_xyz, dtype=float), res_ids, np.asarray(plddt, dtype=float)

def call_chimerax_matchmaker(ref_path, target_path, output_path, cutoff=2.0, debug=False):
    """
    调用 ChimeraX 的 matchmaker 函数进行结构比对
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
        print(f"[ERROR] STDOUT: {e.stdout}")
        print(f"[ERROR] STDERR: {e.stderr}")
        return None
    except FileNotFoundError:
        print("[ERROR] ChimeraX not found in PATH.")
        return None

def compute_properties_with_chimerax(cif_path, temp_dir=None, debug=False):
    """
    使用 ChimeraX (Python 脚本模式) 计算 SASA 和 Secondary Structure (DSSP)。
    返回:
      out_path: 带有 SASA (在 B-factor 中) 的 cif 文件路径
      json_path: 包含残基属性 (seqid, icode, sasa, ss_type) 的 JSON 文件路径
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(dir='.', prefix='sasa_calc_')
        
    base_name = os.path.basename(cif_path)
    root, ext = os.path.splitext(base_name)
    suffix = f"{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}"
    out_path = os.path.join(temp_dir, f"{root}_sasa.{suffix}{ext}")
    json_path = os.path.join(temp_dir, f"{root}_props.{suffix}.json")
    script_path = os.path.join(temp_dir, f"{root}_script.{suffix}.py")

    py_content = f"""
import os
import json
from chimerax.core.commands import run

def process():
    # Open structure
    run(session, 'open "{cif_path}"')
    
    # Measure SASA
    run(session, 'measure sasa #1')
    
    # Compute Secondary Structure (DSSP)
    run(session, 'dssp')
    
    if not session.models:
        print("No models loaded!")
        return

    model = session.models[0]
    
    props = []
    
    # Iterate residues and assign residue area to atom bfactors
    for r in model.residues:
        # SASA
        sasa = getattr(r, 'area', 0.0)
        
        # Secondary Structure
        # ChimeraX residue.ss_type: 0 (Coil/Loop?), 1 (Helix), 2 (Strand)
        ss_type = r.ss_type
        
        # Save props
        props.append({{
            'seqid': r.number,
            'icode': r.insertion_code,
            'name': r.name,
            'sasa': sasa,
            'ss_type': ss_type
        }})
        
        # Assign to all atoms in residue (including CA)
        for a in r.atoms:
            a.bfactor = sasa

    # Save structure with SASA in bfactor
    run(session, 'save "{out_path}" format mmcif models #1')
    
    # Save properties to JSON
    with open("{json_path}", "w") as f:
        json.dump(props, f)
        
    run(session, 'exit')

try:
    process()
except Exception as e:
    print(f"ChimeraX Python Script Error: {{e}}")
    import traceback
    traceback.print_exc()
    import sys
    sys.exit(1)
"""
    
    with open(script_path, "w") as f:
        f.write(py_content)
    
    cmd = [
        'ChimeraX',
        '--nogui',
        '--script', script_path
    ]
    
    try:
        if debug:
            print(f"[DEBUG] Calculating properties for {cif_path} using script {script_path}")
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not os.path.exists(out_path):
            print(f"[WARN] SASA output file {out_path} not found")
            return None, None
        if not os.path.exists(json_path):
            print(f"[WARN] Properties JSON file {json_path} not found")
            return None, None
            
        return out_path, json_path
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Property calculation failed with return code {e.returncode}")
        print(f"[ERROR] STDOUT: {e.stdout}")
        print(f"[ERROR] STDERR: {e.stderr}")
        return None, None
    except Exception as e:
        print(f"[ERROR] Property calculation failed: {e}")
        return None, None

def get_rsa(seq, sasa_values):
    """
    计算相对溶剂可接触表面积 (RSA)
    """
    rsa = []
    for r, s in zip(seq, sasa_values):
        m = MAX_SASA.get(r, 200.0) # 默认 200 如果未知
        if m <= 0: m = 1.0
        rsa.append(s / m)
    return np.array(rsa)

def map_by_geometry_unique(ref_ca, trg_ca, cutoff=2.0):
    """
    一对一几何映射函数
    """
    n_ref, n_trg = len(ref_ca), len(trg_ca)
    d2 = ((ref_ca[:, None, :] - trg_ca[None, :, :]) ** 2).sum(axis=2)
    
    used_trg = set()
    pairs = []
    
    order = np.argsort(d2.min(axis=1))
    
    for i in order:
        j = int(d2[i].argmin())
        if d2[i, j] < (cutoff ** 2) and j not in used_trg:
            pairs.append((int(i), int(j)))
            used_trg.add(j)
    
    return pairs

def compute_mr_cr_for_target(ref_seq, ref_ca, trg_seq, trg_ca, masks, cutoff=2.0):
    """
    计算给定掩码集 (masks) 中每个子集的 MR 和 CR
    masks: dict {name: boolean_array}
    返回: dict {name_MR: float, name_CR: float}
    """
    # 1. 建立所有映射
    pairs = map_by_geometry_unique(ref_ca, trg_ca, cutoff)
    
    results = {}
    
    # 2. 辅助函数：计算特定子集的 MR/CR
    def calc_stats(indices_mask):
        # 筛选出属于该子集（如核心）的映射对
        subset_pairs = [(i, j) for i, j in pairs if indices_mask[i]]
        
        # 分母：该子集在参考结构中的总残基数
        total_ref_class = indices_mask.sum()
        
        mapped_count = len(subset_pairs)
        MR = mapped_count / float(total_ref_class) if total_ref_class > 0 else 0.0
        
        # 分子：在映射对中，氨基酸相同的数量
        conserved_count = sum([ref_seq[i] == trg_seq[j] for i, j in subset_pairs])
        CR = conserved_count / float(mapped_count) if mapped_count > 0 else 0.0
        
        return MR, CR

    for name, mask in masks.items():
        mr, cr = calc_stats(mask)
        results[f"{name}_MR"] = mr
        results[f"{name}_CR"] = cr

    return results

def main():
    ap = argparse.ArgumentParser(description="Compute MR and CR for core/surface and secondary structures using ChimeraX.")
    ap.add_argument("--ref", default=None, help="参考结构的 mmCIF 文件路径")
    ap.add_argument("--targets", nargs="+", default=None, help="待比较的其它物种 mmCIF（一个或多个）")
    ap.add_argument("--root", default=None, help="测试数据根目录，用于自动发现 mmCIF")
    ap.add_argument("--cutoff", type=float, default=2.0, help="Cα 映射阈值（Å）")
    ap.add_argument("--cutoff-sasa", type=float, default=0.25, help="RSA 阈值区分 Core/Surface (默认 0.25)")
    ap.add_argument("--out", default="mr_cr_results.csv", help="输出 CSV 文件")
    ap.add_argument("--temp-dir", default=None, help="临时目录")
    ap.add_argument("--debug", action="store_true", default=False, help="启用调试输出")
    args = ap.parse_args()

    # 处理 ref/targets
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

    # 排除 ref 自身
    ref_real = os.path.realpath(args.ref)
    target_paths = [p for p in sum([glob.glob(t) for t in args.targets], []) if os.path.realpath(p) != ref_real]

    if not target_paths:
        print("[WARN] No targets found.")

    print(f"[INFO] Reference: {args.ref}")
    print(f"[INFO] Targets: {len(target_paths)} files")

    # 创建临时目录
    if args.temp_dir:
        temp_dir = args.temp_dir
    else:
        temp_dir = tempfile.mkdtemp(dir='.', prefix='mr_cr_sasa_')
    os.makedirs(temp_dir, exist_ok=True)

    try:
        # 1. 计算参考结构的 SASA 和 SS
        print("[INFO] Computing SASA and SS for reference...")
        ref_sasa_path, ref_json_path = compute_properties_with_chimerax(args.ref, temp_dir=temp_dir, debug=args.debug)
        if not ref_sasa_path:
            raise RuntimeError("Failed to compute properties for reference.")
        
        # 读取参考结构
        ref_seq, ref_ca, ref_ids, _ = read_cif_structure(ref_sasa_path)
        if len(ref_seq) == 0:
            raise RuntimeError("Reference structure empty or read failed.")
        
        # 读取属性 JSON 并映射到序列
        # 注意：需要通过 res_ids (seqid, icode) 对齐
        with open(ref_json_path, "r") as f:
            props_data = json.load(f)
            
        # 创建属性映射表: (seqid, icode) -> (sasa, ss_type)
        props_map = {}
        for item in props_data:
            # Normalize icode: strip whitespace
            icode = item['icode'].strip()
            key = (item['seqid'], icode)
            props_map[key] = (item['sasa'], item['ss_type'])
            
        # 构建对齐后的 sasa 和 ss 数组
        sasa_aligned = []
        ss_aligned = []
        
        for rid in ref_ids:
            # Normalize rid icode
            rid_key = (rid[0], rid[1].strip())
            
            if rid_key in props_map:
                s, ss = props_map[rid_key]
                sasa_aligned.append(s)
                ss_aligned.append(ss)
            else:
                if args.debug:
                    print(f"[WARN] Residue {rid} (key={rid_key}) not found in props_map")
                sasa_aligned.append(0.0)
                ss_aligned.append(0) # Default Coil?
                
        sasa_aligned = np.array(sasa_aligned)
        ss_aligned = np.array(ss_aligned)
        
        # 计算 RSA 并划分 Core/Surface
        ref_rsa = get_rsa(ref_seq, sasa_aligned)
        
        # 创建各种 Mask
        masks = {
            'core': ref_rsa < args.cutoff_sasa,
            'surface': ref_rsa >= args.cutoff_sasa,
            'helical': ss_aligned == 1,
            'extended': ss_aligned == 2,
            'coil': (ss_aligned != 1) & (ss_aligned != 2) # Assume 0 and others are coil/loop
        }
        
        print(f"[INFO] Ref Length: {len(ref_seq)}")
        for k, v in masks.items():
            count = v.sum()
            print(f"[INFO] {k.capitalize()} Residues: {count}")
            if count == 0:
                print(f"[WARN] Reference structure has 0 residues classified as '{k}'. MR/CR for this category will be 0 for all targets.")
        
        # 准备 CSV 表头
        headers = ["ref", "target", "target_file"]
        for k in masks.keys():
            headers.extend([f"{k}_MR", f"{k}_CR"])
            
        rows = []
        rows.append(headers)
        
        for tf in target_paths:
            try:
                if args.debug:
                    print(f"[PROCESSING] {os.path.basename(tf)}")
                
                # 2. 对齐 Target 到 Reference
                aligned_path = os.path.join(temp_dir, f"aligned_{os.path.basename(tf)}")
                final_aligned = call_chimerax_matchmaker(args.ref, tf, aligned_path, cutoff=args.cutoff, debug=args.debug)
                
                if not final_aligned:
                    raise RuntimeError("Alignment failed")
                
                # 3. 读取对齐后的 Target
                trg_seq, trg_ca, _, _ = read_cif_structure(final_aligned)
                
                # 4. 计算 MR/CR
                res_dict = compute_mr_cr_for_target(
                    ref_seq, ref_ca, trg_seq, trg_ca, masks, cutoff=args.cutoff
                )
                
                row = [extract_protein_name(args.ref), extract_protein_name(tf), os.path.basename(tf)]
                for k in masks.keys():
                    row.append(f"{res_dict.get(f'{k}_MR', 0.0):.4f}")
                    row.append(f"{res_dict.get(f'{k}_CR', 0.0):.4f}")
                
                rows.append(row)
                
            except Exception as e:
                # Fill NA
                row = [extract_protein_name(args.ref), extract_protein_name(tf), os.path.basename(tf)]
                row.extend(["NA"] * (len(headers) - 3))
                rows.append(row)
                sys.stderr.write(f"[ERROR] {tf}: {e}\n")
        
        # 保存结果
        with open(args.out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"[OK] Wrote: {args.out}")

    finally:
        # 清理临时目录
        if not args.debug and args.temp_dir is None:
            shutil.rmtree(temp_dir)
            pass

if __name__ == "__main__":
    main()
