#!/bin/bash
# mrcr_one.sh — 单个 target 的 compute worker
# 由 run_mrcr.sh / run_mrcr_structure.sh 用 xargs 调用
# 用法: bash mrcr_one.sh compute|structure <module> <target> <tier> <ref_wp> <ref_path>

set -u
MODE="$1"; MODULE="$2"; TARGET="$3"; TIER="$4"; REF_WP="$5"; REF_PATH="$6"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 conda env (xargs subshell 不继承 activate)
if [[ -z "${CONDA_MRCR_ACTIVATED:-}" ]]; then
    source ~/mambaforge/etc/profile.d/conda.sh
    conda activate mrcr
    export CONDA_MRCR_ACTIVATED=1
fi

# 选脚本 + 输出目录
if [[ "$MODE" == "structure" ]]; then
    SCRIPT="$HERE/structure_mr_cr_ChimeraX.py"
    OUT="$HERE/$MODULE/Result_Structure/${TARGET}_mrcr_results.csv"
    ERRLOG="$HERE/run_mrcr_structure.err"
else
    SCRIPT="$HERE/compute_mr_cr_ChimeraX.py"
    OUT="$HERE/$MODULE/Result/${TARGET}_mrcr_results.csv"
    ERRLOG="$HERE/run_mrcr.err"
fi

if [[ -s "$OUT" ]]; then
    echo "[SKIP] $MODE $MODULE/$TARGET" >&2
    exit 0
fi
if [[ ! -f "$REF_PATH" ]]; then
    echo "[WARN] $MODE $MODULE/$TARGET ref missing: $REF_PATH" >&2
    exit 0
fi

GLOB="$HERE/$MODULE/$TARGET/*/*/seed-1_sample-0/model.cif"
echo "[RUN ] $MODE $MODULE/$TARGET tier=$TIER ref=$REF_WP" >&2

# --exclude-self 只在 compute 模式支持，structure 脚本内部已自动排除
if [[ "$MODE" == "structure" ]]; then
    python3 "$SCRIPT" \
        --ref "$REF_PATH" \
        --targets "$GLOB" \
        --out "$OUT" 2>>"$ERRLOG" || {
            echo "[FAIL] $MODE $MODULE/$TARGET" >&2
            exit 1
        }
else
    python3 "$SCRIPT" \
        --ref "$REF_PATH" \
        --targets "$GLOB" \
        --exclude-self \
        --out "$OUT" 2>>"$ERRLOG" || {
            echo "[FAIL] $MODE $MODULE/$TARGET" >&2
            exit 1
        }
fi
