#!/usr/bin/env bash
# Pod environment. A venv that inherits the image's PyTorch 2.8 / CUDA 12.9 stack
# (nvcr.io/nvidia/pytorch:25.06) plus the few missing packages and the Rust extension.
# Downloading the locked torch wheels on the pod is too slow, and Phase 2 (mamba-ssm) is
# planned against this same PyTorch 2.8 build.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/home/data/.local/bin:/home/data/.cargo/bin:$PATH
export CARGO_HOME=${CARGO_HOME:-/home/data/.cargo} RUSTUP_HOME=${RUSTUP_HOME:-/home/data/.rustup}
[ -x .venv/bin/python ] || uv venv --system-site-packages --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python loguru h5py
uv pip install --python .venv/bin/python --no-deps --reinstall ./rust/dlfast
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -c "import torch, dlfast, dl_final_prj; print(torch.__version__, torch.cuda.is_available())"
