"""SpaceGen AI: constraint-aware interior layout generation (see Tech_Spec.md)."""
import os
import sys

# Deterministic cuBLAS needs this variable before CUDA creates its first cuBLAS handle,
# which happens lazily at the first GPU matrix operation. Importing spacegen before any
# CUDA work is therefore enough. If CUDA was already running, the variable comes too late
# to take effect, and seed.set_seed() refuses to claim determinism.
_torch = sys.modules.get("torch")
CUBLAS_SET_TOO_LATE = (
    "CUBLAS_WORKSPACE_CONFIG" not in os.environ
    and _torch is not None
    and _torch.cuda.is_initialized()
)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
