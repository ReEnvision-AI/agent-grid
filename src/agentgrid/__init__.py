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


env = Env()
env.read_env(override=True)

__version__ = os.getenv("AGENT_GRID_VERSION", "1.1.2")


if not os.getenv("AGENT_GRID_IGNORE_DEPENDENCY_VERSION"):
    assert (
        version.parse("4.43.1") <= version.parse(transformers.__version__) <= version.parse("4.54.1")
    ), "Please install a proper transformers version: pip install transformers>=4.43.1,<= 4.54.1"


def _override_bfloat16_mode_default():
    if os.getenv("USE_LEGACY_BFLOAT16") is None:
        hivemind.compression.base.USE_LEGACY_BFLOAT16 = False


_initialize_logs()
_override_bfloat16_mode_default()
