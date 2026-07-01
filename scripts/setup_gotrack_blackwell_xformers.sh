#!/usr/bin/env bash
# Rebuild xformers in the gotrack_cu128 env for Blackwell / SM120 GPUs.
#
# The cu128 xformers wheel can import successfully while GoTrack's DINOv2 path
# still fails on Blackwell if it feeds FP32 tensors into xformers:
#   No operator found for memory_efficient_attention_forward
#
# xformers' Blackwell attention kernels support FP16/BF16, not FP32. This
# script fixes the environment and verifies the BF16 xformers path instead of
# disabling xformers in code.
set -euo pipefail

ENV_NAME="${ENV_NAME:-gotrack_cu128}"
CONDA_DIR="${CONDA_DIR:-$HOME/anaconda3}"
PY="${PY:-$CONDA_DIR/envs/$ENV_NAME/bin/python}"
XFORMERS_REF="${XFORMERS_REF:-v0.0.35}"
CUDA_TOOLKIT_VERSION="${CUDA_TOOLKIT_VERSION:-12.8.1}"
CUDA_NVCC_VERSION="${CUDA_NVCC_VERSION:-12.8.93}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
MAX_JOBS="${MAX_JOBS:-8}"
VERIFY_ONLY=0

usage() {
    cat <<EOF
usage: $0 [--verify-only]

Environment overrides:
  ENV_NAME                 Conda env name. Default: gotrack_cu128
  CONDA_DIR                Conda root. Default: \$HOME/anaconda3
  XFORMERS_REF             xformers git ref to build. Default: v0.0.35
  TORCH_CUDA_ARCH_LIST     CUDA arch list for source build. Default: 12.0
  MAX_JOBS                 Parallel build jobs. Default: 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verify-only)
            VERIFY_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$PY" ]]; then
    echo "[setup] missing python interpreter: $PY" >&2
    exit 1
fi

echo "[setup] host=$(hostname)"
echo "[setup] env=$ENV_NAME"
echo "[setup] python=$PY"
echo "[setup] xformers_ref=$XFORMERS_REF"
echo "[setup] torch_cuda_arch_list=$TORCH_CUDA_ARCH_LIST"
echo "[setup] max_jobs=$MAX_JOBS"

source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST
export MAX_JOBS

diagnose() {
    "$PY" - <<'PY'
import os
import shutil
import subprocess
import sys

import torch
from torch.utils.cpp_extension import CUDA_HOME

print("[diag] executable", sys.executable, flush=True)
print("[diag] torch", torch.__version__, flush=True)
print("[diag] torch_cuda", torch.version.cuda, flush=True)
print("[diag] cuda_available", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("[diag] gpu", torch.cuda.get_device_name(0), flush=True)
    print("[diag] capability", torch.cuda.get_device_capability(0), flush=True)
print("[diag] cuda_home", CUDA_HOME, flush=True)
print("[diag] nvcc_path", shutil.which("nvcc"), flush=True)
try:
    import xformers
    print("[diag] xformers", xformers.__version__, flush=True)
except Exception as exc:
    print("[diag] xformers_import_error", repr(exc), flush=True)

try:
    result = subprocess.run(
        [sys.executable, "-m", "xformers.info"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for line in result.stdout.splitlines():
        if (
            "memory_efficient_attention" in line
            or "build.torch_version" in line
            or "build.env.TORCH_CUDA_ARCH_LIST" in line
            or "build.cuda_version" in line
            or "build.nvcc_version" in line
            or "gpu.compute_capability" in line
            or "gpu.name" in line
        ):
            print("[xformers.info]", line, flush=True)
except Exception as exc:
    print("[diag] xformers_info_error", repr(exc), flush=True)
PY
}

verify_attention() {
    "$PY" - <<'PY'
import torch
from xformers.ops import memory_efficient_attention

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

torch.cuda.empty_cache()
q = torch.randn((48, 405, 6, 64), device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
y = memory_efficient_attention(q, k, v)
torch.cuda.synchronize()
print("[verify] memory_efficient_attention_bf16_ok", tuple(y.shape), y.dtype, flush=True)
PY
}

diagnose

if [[ "$VERIFY_ONLY" -eq 1 ]]; then
    verify_attention
    echo "[setup] verify-only complete"
    exit 0
fi

echo "[setup] installing build dependencies"
"$PY" -m pip install --upgrade "setuptools<82" wheel packaging ninja cmake pybind11
"$PY" -m pip install --upgrade \
    "cuda-toolkit[cublas,cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==$CUDA_TOOLKIT_VERSION" \
    "nvidia-cuda-nvcc-cu12==$CUDA_NVCC_VERSION"

echo "[setup] rebuilding xformers from source for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
"$PY" -m pip install \
    --force-reinstall \
    --no-build-isolation \
    --no-cache-dir \
    --no-deps \
    -v \
    "git+https://github.com/facebookresearch/xformers.git@$XFORMERS_REF"

echo "[setup] post-install diagnostics"
diagnose
verify_attention

echo "[setup] done"
