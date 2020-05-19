# -*- coding: utf-8 -*-

"""
This module contains a Gym wrapper that repeats the same action until the observation does not change.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any

from gym_breakout_pygame.breakout_env import Breakout, BreakoutConfiguration


class BreakoutSkipper(Breakout, ABC):
    """Repeat same step until a different observation is obtained."""

    def __init__(self, breakout_config: Optional[BreakoutConfiguration] = None):
        super().__init__(breakout_config)
        self._previous_obs = None  # type: Any

    @classmethod
    @abstractmethod
    def compare(cls, obs1, obs2):
        """Compare two observations"""
        return False

    def reset(self):
        obs = super().reset()
        self._previous_obs = obs
        return obs

    def step(self, action: int):
        obs, reward, is_finished, info = super().step(action)
        while self.compare(obs, self._previous_obs) and not is_finished:
            next_obs, next_reward, next_is_finished, next_info = super().step(action)
            obs = next_obs
            reward += next_reward
            is_finished = is_finished or next_is_finished
            info.update(next_info)

        self._previous_obs = obs
        return obs, reward, is_finished, info

