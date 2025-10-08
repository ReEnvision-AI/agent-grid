#
# Copyright (c) 2025 ReEnvision AI, LLC. All rights reserved.
#
# This software is the confidential and proprietary information of
# ReEnvision AI, LLC ("Confidential Information"). You shall not
# disclose such Confidential Information and shall use it only in
# accordance with the terms of the license agreement you entered into
# with ReEnvision AI, LLC.
#

from abc import ABC, abstractmethod

import torch


class TaskPrioritizerBase(ABC):
    """Abstract class for TaskPrioritizer whose responsibility is to evaluate task priority"""

    @abstractmethod
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        """Evaluates task value by the amount of points given, task input and additional kwargs. Lower priority is better"""
        pass


class DummyTaskPrioritizer(TaskPrioritizerBase):
    def prioritize(self, *input: torch.Tensor, points: float = 0.0, **kwargs) -> float:
        # Inference steps go first since they are more latency-sensitive
        if kwargs.get("type") == "inference":
            return 1.0
        return 2.0  # Forward, backward
