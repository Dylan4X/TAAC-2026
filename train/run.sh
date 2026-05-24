#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

: "${TAAC_INSTALL_FLASH_ATTN:=1}"
: "${TAAC_ATTN_BACKEND:=flash_attn}"
: "${TAAC_VALID_RATIO:=0.02}"
: "${TAAC_NUM_WORKERS:=8}"
: "${TAAC_AMP_MODE:=bf16}"
: "${TAAC_ENABLE_TF32:=1}"
: "${TAAC_CUDNN_BENCHMARK:=1}"
: "${TAAC_DETERMINISTIC:=0}"
: "${TAAC_MATMUL_PRECISION:=high}"
: "${TAAC_TRAIN_PREFETCH_FACTOR:=6}"
: "${TAAC_VALID_NUM_WORKERS:=4}"
: "${TAAC_VALID_PREFETCH_FACTOR:=4}"
: "${TAAC_GRAD_CLIP_NORM:=1.0}"
: "${TAAC_WRITER_INTERVAL:=100}"
: "${TAAC_SHOW_PROGRESS:=0}"
: "${TAAC_EMPTY_CACHE_AFTER_EVAL:=0}"
: "${TAAC_USE_META_QUERY_CONDITIONING:=1}"
: "${TAAC_META_QUERY_GATE_INIT:=0.01}"
: "${TAAC_LOSS_TYPE:=bce_pairwise}"
: "${TAAC_PAIRWISE_LAMBDA:=0.05}"
: "${TAAC_WARMUP_STEPS:=500}"
: "${TAAC_LR_SCHEDULE:=cosine}"
: "${TAAC_EMA_DECAY:=0.999}"
: "${TAAC_LABEL_SMOOTHING:=0.01}"
: "${TAAC_WEIGHT_DECAY:=0.02}"
: "${OMP_NUM_THREADS:=1}"
: "${MKL_NUM_THREADS:=1}"
: "${OPENBLAS_NUM_THREADS:=1}"
: "${NUMEXPR_NUM_THREADS:=1}"

CACHE_ROOT="${USER_CACHE_PATH:-${SCRIPT_DIR}/.cache}"
SITE_PACKAGES_DIR="${CACHE_ROOT}/taac_flash_attn_site_packages"
WHEEL_SUBDIR="${SCRIPT_DIR}/wheels"
FA_WHEEL_CACHE_DIR="${CACHE_ROOT}/flash_attn_cache"
FA_SOURCE_EXTRACT_DIR="${CACHE_ROOT}/flash_attn_source_build/train_v11"

export TAAC_ATTN_BACKEND="${TAAC_ATTN_BACKEND}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS

mkdir -p "${CACHE_ROOT}" "${SITE_PACKAGES_DIR}" "${FA_WHEEL_CACHE_DIR}"

find_first_match() {
    local search_dir="$1"
    local pattern="$2"
    if [ -d "${search_dir}" ]; then
        find "${search_dir}" -maxdepth 1 -type f -name "${pattern}" 2>/dev/null | sort | head -1
    fi
}

if [ "${TAAC_INSTALL_FLASH_ATTN}" = "1" ]; then
    FA_CACHED="$(find_first_match "${FA_WHEEL_CACHE_DIR}" 'flash_attn*.whl')"

    FA_BUNDLED="$(find_first_match "${WHEEL_SUBDIR}" 'flash_attn*.whl')"
    if [ -z "${FA_BUNDLED}" ]; then
        FA_BUNDLED="$(find_first_match "${SCRIPT_DIR}" 'flash_attn*.whl')"
    fi

    if [ -z "${FA_CACHED}" ] && [ -z "${FA_BUNDLED}" ]; then
        FA_SOURCE_ARCHIVE="$(find_first_match "${SCRIPT_DIR}" 'flash_attn*.tar.gz')"
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${SCRIPT_DIR}" 'flash_attn*.tgz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${SCRIPT_DIR}" 'flash-attention*.zip')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${SCRIPT_DIR}" 'flash-attention*.tar.gz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${SCRIPT_DIR}" 'flash-attention*.tgz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${WHEEL_SUBDIR}" 'flash_attn*.tar.gz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${WHEEL_SUBDIR}" 'flash_attn*.tgz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${WHEEL_SUBDIR}" 'flash-attention*.zip')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${WHEEL_SUBDIR}" 'flash-attention*.tar.gz')"
        fi
        if [ -z "${FA_SOURCE_ARCHIVE}" ]; then
            FA_SOURCE_ARCHIVE="$(find_first_match "${WHEEL_SUBDIR}" 'flash-attention*.tgz')"
        fi

        if [ -n "${FA_SOURCE_ARCHIVE}" ]; then
            echo "Extracting flash-attn source archive ${FA_SOURCE_ARCHIVE} into ${FA_SOURCE_EXTRACT_DIR}"
            FA_SOURCE_DIR="$(
                python3 - "${FA_SOURCE_ARCHIVE}" "${FA_SOURCE_EXTRACT_DIR}" <<'PY'
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

archive = Path(sys.argv[1])
dest = Path(sys.argv[2])
if dest.exists():
    shutil.rmtree(dest)
dest.mkdir(parents=True, exist_ok=True)

name = archive.name.lower()
if name.endswith(".zip"):
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
elif name.endswith(".tar.gz") or name.endswith(".tgz"):
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(dest)
else:
    raise SystemExit("")

candidates = [p for p in dest.iterdir() if p.is_dir() and (p / "setup.py").exists()]
if not candidates:
    raise SystemExit("")
print(sorted(candidates)[0])
PY
            )"

            if [ -n "${FA_SOURCE_DIR}" ]; then
                echo "Building flash-attn wheel from source tree ${FA_SOURCE_DIR} into ${FA_WHEEL_CACHE_DIR} (first run only) ..."
                python3 -m pip wheel \
                    --no-build-isolation \
                    --no-deps \
                    "${FA_SOURCE_DIR}" \
                    -w "${FA_WHEEL_CACHE_DIR}" \
                    2>&1 || echo "flash-attn source build failed; attention will fall back to SDPA."
                FA_CACHED="$(find_first_match "${FA_WHEEL_CACHE_DIR}" 'flash_attn*.whl')"
            else
                echo "flash-attn source archive was found but no buildable source tree was detected; attention will fall back to SDPA."
            fi
        else
            echo "No flash-attn source archive found locally; attention will fall back to SDPA."
        fi
    fi

    INSTALL_STAMP="${SITE_PACKAGES_DIR}/.flash_attn_install_complete"
    SELECTED_WHEEL="${FA_CACHED:-${FA_BUNDLED:-}}"
    if [ -n "${SELECTED_WHEEL}" ]; then
        NEED_INSTALL=0
        if [ ! -f "${INSTALL_STAMP}" ] || [ "${SELECTED_WHEEL}" -nt "${INSTALL_STAMP}" ]; then
            NEED_INSTALL=1
        fi
        if [ "${NEED_INSTALL}" = "1" ]; then
            echo "Installing flash-attn wheel into ${SITE_PACKAGES_DIR}"
            if python3 -m pip install \
                --target "${SITE_PACKAGES_DIR}" \
                --no-compile \
                --no-deps \
                --disable-pip-version-check \
                "${SELECTED_WHEEL}"; then
                touch "${INSTALL_STAMP}"
            else
                echo "flash-attn install failed; attention will fall back to SDPA."
            fi
        else
            echo "Reusing shared flash-attn install from ${SITE_PACKAGES_DIR}"
        fi
        export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH:-}"
    fi
fi

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --valid_ratio "${TAAC_VALID_RATIO}" \
    --use_meta_query_conditioning "${TAAC_USE_META_QUERY_CONDITIONING}" \
    --meta_query_gate_init "${TAAC_META_QUERY_GATE_INIT}" \
    --loss_type "${TAAC_LOSS_TYPE}" \
    --pairwise_lambda "${TAAC_PAIRWISE_LAMBDA}" \
    --num_workers "${TAAC_NUM_WORKERS}" \
    --amp_mode "${TAAC_AMP_MODE}" \
    --enable_tf32 "${TAAC_ENABLE_TF32}" \
    --cudnn_benchmark "${TAAC_CUDNN_BENCHMARK}" \
    --deterministic "${TAAC_DETERMINISTIC}" \
    --matmul_precision "${TAAC_MATMUL_PRECISION}" \
    --train_prefetch_factor "${TAAC_TRAIN_PREFETCH_FACTOR}" \
    --valid_num_workers "${TAAC_VALID_NUM_WORKERS}" \
    --valid_prefetch_factor "${TAAC_VALID_PREFETCH_FACTOR}" \
    --grad_clip_norm "${TAAC_GRAD_CLIP_NORM}" \
    --writer_interval "${TAAC_WRITER_INTERVAL}" \
    --show_progress "${TAAC_SHOW_PROGRESS}" \
    --empty_cache_after_eval "${TAAC_EMPTY_CACHE_AFTER_EVAL}" \
    --warmup_steps "${TAAC_WARMUP_STEPS}" \
    --lr_schedule "${TAAC_LR_SCHEDULE}" \
    --ema_decay "${TAAC_EMA_DECAY}" \
    --label_smoothing "${TAAC_LABEL_SMOOTHING}" \
    --weight_decay "${TAAC_WEIGHT_DECAY}" \
    "$@"
