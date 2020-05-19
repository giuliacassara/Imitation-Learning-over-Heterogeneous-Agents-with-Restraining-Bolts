import gym
import numpy as np
from flloat.semantics import PLInterpretation
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from gym_breakout_pygame.wrappers.dict_space import BreakoutDictSpace


class BreakoutWrapper(gym.ObservationWrapper):

    def __init__(self, config: BreakoutConfiguration):
        super().__init__(BreakoutDictSpace(config))
        self._previous_brick_matrix = None
        self._next_brick_matrix = None

    def observation(self, observation):
        new_observation = observation
        new_observation["previous_bricks_matrix"] = self._previous_brick_matrix
        self._previous_brick_matrix = np.copy(self._next_brick_matrix)
        self._next_brick_matrix = new_observation["bricks_matrix"]
        return new_observation

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._previous_brick_matrix = np.copy(obs["bricks_matrix"])
        self._next_brick_matrix = self._previous_brick_matrix
        return obs


def make_goal(nb_columns: int = 3) -> str:
    """
    Define the goal expressed in LDLf logic.

    E.g. for nb_columns = 3:

        <(!c0 & !c1 & !c2)*;c0;(!c0 & !c1 & !c2)*;c1;(!c0 & !c1 & !c2)*;c2>tt

    :param nb_columns: the number of column
    :return: the string associated with the goal.
    """
    labels = ["c" + str(column_id) for column_id in range(nb_columns)]
    empty = "(!" + " & !".join(labels) + ")"
    f = "<" + empty + "*;{}>tt"
    regexp = (";" + empty + "*;").join(labels)
    f = f.format(regexp)
    return f


def extract_breakout_fluents(obs, action) -> PLInterpretation:
    brick_matrix = obs["bricks_matrix"]  # type: np.ndarray
    previous_brick_matrix = obs["previous_bricks_matrix"]  # type: np.ndarray
    previous_broken_columns = np.all(previous_brick_matrix == 0.0, axis=1)
    current_broken_columns = np.all(brick_matrix == 0.0, axis=1)
    compare = (previous_broken_columns == current_broken_columns)  # type: np.ndarray
    if compare.all():
        result = PLInterpretation(set())
        return result
    else:
        index = np.argmin(compare)
        fluent = "c" + str(index)
        result = PLInterpretation({fluent})
        return result
