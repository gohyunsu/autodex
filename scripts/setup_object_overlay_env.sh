#!/usr/bin/env bash
# Prepare the conda env used by overlay_object_video_single.py.
#
# Use the Blackwell-compatible GoTrack env by default. The older paradex env can
# import the overlay dependencies, but its torch build may not support SM120.
set -euo pipefail

ENV_NAME="${ENV_NAME:-gotrack_cu128}"
CONDA_DIR="${CONDA_DIR:-$HOME/anaconda3}"
PY="${PY:-$CONDA_DIR/envs/$ENV_NAME/bin/python}"

if [[ ! -x "$PY" ]]; then
    echo "[overlay-env] missing python interpreter: $PY" >&2
    exit 1
fi

echo "[overlay-env] host=$(hostname)"
echo "[overlay-env] python=$PY"

"$PY" -m pip install opencv-python transforms3d trimesh open3d

if ! "$PY" - <<'PY'
import nvdiffrast.torch as dr
print("nvdiffrast_ok")
PY
then
    "$PY" -m pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
fi

"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "paradex"))

import cv2
import nvdiffrast.torch as dr
import torch
import transforms3d
import trimesh
import open3d
from paradex.image.projection import intr_opencv_to_opengl_proj

if not torch.cuda.is_available():
    raise SystemExit("cuda_not_available")
device = torch.device("cuda")
x = torch.eye(4, device=device)
_ = x @ x
print("torch_cuda_ok", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print("overlay_imports_ok")
PY

echo "[overlay-env] done"
