# -*- coding: utf-8 -*-

"""Breakout environments using a "normal" state space.
- BreakoutNMultiDiscrete
- BreakoutNDiscrete
"""

from typing import Optional

import numpy as np
from gym.spaces import Discrete, MultiDiscrete

from gym_breakout_pygame.breakout_env import Breakout, BreakoutConfiguration, BreakoutState
from gym_breakout_pygame.utils import encode
from gym_breakout_pygame.wrappers.skipper import BreakoutSkipper


class BreakoutNMultiDiscrete(BreakoutSkipper):
    """
    Breakout env with a gym.MultiDiscrete observation space composed by:
    - paddle x position
    - ball x position
    - ball y position
    - ball direction

    """

    def __init__(self, config: Optional[BreakoutConfiguration] = None):
        super().__init__(config)
        self.observation_space = MultiDiscrete((
            self._paddle_x_space.n,
            self._ball_x_space.n,
            self._ball_y_space.n,
            self._ball_x_speed_space.n,
            self._ball_y_speed_space.n
        ))

    @classmethod
    def compare(cls, obs1: np.ndarray, obs2: np.ndarray):
        return (obs1 == obs2).all()

    @classmethod
    def observe(cls, state: BreakoutState):
        paddle_x = state.paddle.x // state.config.resolution_x
        ball_x = state.ball.x // state.config.resolution_x
        ball_y = state.ball.y // state.config.resolution_y
        ball_x_speed = state.ball.speed_x_norm
        ball_y_speed = state.ball.speed_y_norm

        obs = [paddle_x, ball_x, ball_y, ball_x_speed, ball_y_speed]
        return np.asarray(obs)


class BreakoutNDiscrete(BreakoutSkipper):
    """
    The same of BreakoutNMultiDiscrete, but the observation space encoded in one integer.
    """

    def __init__(self, config: Optional[BreakoutConfiguration] = None):
        super().__init__(config)
        self.observation_space = Discrete(config.n_paddle_x * config.n_ball_x * config.n_ball_y
                                          * config.n_ball_x_speed * config.n_ball_y_speed)

    @classmethod
    def observe(cls, state: BreakoutState):
        obs = BreakoutNMultiDiscrete.observe(state)
        dims = [state.config.n_paddle_x, state.config.n_ball_x, state.config.n_ball_y,
                state.config.n_ball_x_speed, state.config.n_ball_y_speed]
        return encode(list(obs), dims)

    @classmethod
    def compare(cls, obs1, obs2):
        return obs1 == obs2
