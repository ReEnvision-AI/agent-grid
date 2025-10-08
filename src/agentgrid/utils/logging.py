#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

import os

from hivemind.utils import logging as hm_logging


def initialize_logs():
    """Initialize Agent Grid logging tweaks. This function is called when you import the `agentgrid` module."""

    # Env var AGENT_GRID_LOGGING=False prohibits AGent Grid do anything with logs
    if os.getenv("AGENT_GRID_LOGGING", "True").lower() in ("false", "0"):
        return

    hm_logging.use_hivemind_log_handler("in_root_logger")

    # We suppress asyncio error logs by default since they are mostly not relevant for the end user,
    # unless there is env var AGENT_GRID_ASYNCIO_LOGLEVEL
    asyncio_loglevel = os.getenv("AGENT_GRID_ASYNCIO_LOGLEVEL", "FATAL" if hm_logging.loglevel != "DEBUG" else "DEBUG")
    hm_logging.get_logger("asyncio").setLevel(asyncio_loglevel)
