#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#


import torch
from hivemind.utils import get_logger

logger = get_logger(__name__)

PUBLIC_INITIAL_PEERS = ['/dns4/sociallyshaped.net/tcp/8788/p2p/QmSt3bPSboHuBNfgB3tPrjGnW1D3xFRPyvrmi2x7TiZ3qR', '/ip4/52.14.122.164/tcp/8788/p2p/QmT5mCzypk1HwyEaZ9JbKRoypG235i5KghUoFo32VDaTEZ']


# The reachability API is currently used only when connecting to the public swarm
REACHABILITY_API_URL = "https://sociallyshaped.net/health"

DTYPE_MAP = dict(bfloat16=torch.bfloat16, float16=torch.float16, float32=torch.float32, auto="auto")
