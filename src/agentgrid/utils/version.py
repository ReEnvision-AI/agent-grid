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
import re
from typing import Union

import requests
from hivemind.utils.logging import TextStyle, get_logger
from packaging.version import parse

import agentgrid

logger = get_logger(__name__)


def validate_version() -> None:
    ...


def get_compatible_model_repo(model_name_or_path: Union[str, os.PathLike, None]) -> Union[str, os.PathLike, None]:
    ...
