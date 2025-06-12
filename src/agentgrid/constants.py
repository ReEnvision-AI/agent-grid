
import torch
from hivemind.utils import get_logger

logger = get_logger(__name__)

PUBLIC_INITIAL_PEERS = ['/dns4/sociallyshaped.net/tcp/8788/p2p/QmTUpY86VSyvwvBN8oc9W3JztLaxyabT6b17gnXxdfx5HL']

# The reachability API is currently used only when connecting to the public swarm
REACHABILITY_API_URL = "https://sociallyshaped.net/health"

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
