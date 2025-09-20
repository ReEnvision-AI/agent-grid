import os
import platform
from environs import Env

os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

if platform.system() == "Darwin":
    # Necessary for forks to work properly on macOS, see https://github.com/kevlened/pytest-parallel/issues/93
    os.environ.setdefault("no_proxy", "*")
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

import hivemind
import transformers
from packaging import version

from agentgrid.client import *
from agentgrid.models import *
from agentgrid.utils import *
from agentgrid.utils.logging import initialize_logs as _initialize_logs

try:  # enable faster matmul kernels when available
    import torch

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            sdp = torch.backends.cuda.sdp_kernel
            sdp.enable_flash_sdp(True)
            sdp.enable_math_sdp(True)
            sdp.enable_mem_efficient_sdp(True)
        except Exception:  # pragma: no cover
            pass
except Exception:  # pragma: no cover - best effort tuning
    pass


env = Env()
env.read_env(override=True)

# Read version from VERSION file
with open(os.path.join(os.path.dirname(__file__), "VERSION"), "r") as f:
    __version__ = f.read().strip()


if not os.getenv("AGENT_GRID_IGNORE_DEPENDENCY_VERSION"):
    assert (
        version.parse("4.43.1") <= version.parse(transformers.__version__) <= version.parse("4.55.4")
    ), "Please install a proper transformers version: pip install transformers>=4.43.1,<= 4.55.4"


def _override_bfloat16_mode_default():
    if os.getenv("USE_LEGACY_BFLOAT16") is None:
        hivemind.compression.base.USE_LEGACY_BFLOAT16 = False


_initialize_logs()
_override_bfloat16_mode_default()
