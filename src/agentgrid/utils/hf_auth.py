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
from typing import Union


def always_needs_auth(model_name: Union[str, os.PathLike, None]) -> bool:
    loading_from_repo = model_name is not None and not os.path.isdir(model_name)
    return loading_from_repo and model_name.startswith("meta-llama/Llama")
