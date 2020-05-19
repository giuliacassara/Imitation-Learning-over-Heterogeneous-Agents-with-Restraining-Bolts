import os
from pathlib import Path

import gym
import pythomata
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym.spaces import MultiDiscrete
from gym_sapientino.sapientino_env import SapientinoConfiguration, color2int
from gym_sapientino.wrappers.dict_space import SapientinoDictSpace
from pythomata.dfa import DFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper

from rl_algorithm.temporal import TemporalGoalWrapperLogTraces

colors = [c.value for c in color2int][1:4]


class SapientinoWrapper(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = MultiDiscrete((
            self.unwrapped._x_space.n,
            self.unwrapped._y_space.n,
            self.unwrapped._theta_space.n
        ))


class SapientinoTemporalWrapper(TemporalGoalWrapper):


    def step(self, action):
        obs, reward, done, info = super().step(action)
        some_failed = any(tg.is_failed() for tg in self.temp_goals)
        all_true = all(tg.is_true() for tg in self.temp_goals)
        new_done = done or some_failed or all_true
        return obs, reward, new_done, info


def extract_sapientino_fluents(obs, action) -> PLInterpretation:
    color_idx = obs["color"]
    beep = obs["beep"]

    fluents = set()

    if 0 < color_idx <= len(colors) and beep:
        color_string = colors[color_idx - 1]
        fluents.add(color_string)
    elif color_idx == 0 and beep:
        fluents.add("bad_beep")

    result = PLInterpretation(fluents)
    return result


def make_goal() -> str:
    """
    Define the goal for Sapientino.

    :return: the string associated with the goal.
    """
    labels = [color for color in colors]
    empty = "!bad_beep & !" + " & !".join(labels)
    f = "<(" + empty + ")*;{}>tt"
    regexp = (";(" + empty + ")*;").join(labels)
    f = f.format(regexp)

    return f


def make_env(config: SapientinoConfiguration, output_dir, goal_reward: float = 1000.0,
             reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param output_dir: the path to the output directory.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """

    formula_string = make_goal()
    print("Formula: {}".format(formula_string))
    formula = LDLfParser()(formula_string)
    tg = TemporalGoal(formula=formula,
                      reward=goal_reward,
                      labels={color for color in colors}.union({"bad_beep"}),
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_sapientino_fluents)

    tg._automaton.to_dot(os.path.join(output_dir, "true_automaton"))
    print("Original automaton at {}".format(os.path.join(output_dir, "true_automaton.svg")))

    env = SapientinoTemporalWrapper(
        SapientinoWrapper(SapientinoDictSpace(config)),
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=lambda obs, action:
            (obs["x"], obs["y"], obs["th"]) if config.differential else (obs["x"], obs["y"])
    )

    positive_traces_path = Path(output_dir, "positive_traces.txt")
    negative_traces_path = Path(output_dir, "negative_traces.txt")
    env = TemporalGoalWrapperLogTraces(env, extract_sapientino_fluents, positive_traces_path, negative_traces_path)

    return env


def make_env_from_dfa(config: SapientinoConfiguration, dfa: pythomata.dfa.DFA,
                      goal_reward: float = 1000.0,  reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param dfa: the automaton that constitutes the goal.
    :param goal_reward: the reward associated to the goal.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    tg = TemporalGoal(automaton=dfa,
                      reward=goal_reward,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_sapientino_fluents)

    env = SapientinoTemporalWrapper(
        SapientinoWrapper(SapientinoDictSpace(config)),
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=lambda obs, action:
            (obs["x"], obs["y"], obs["theta"]) if config.differential else (obs["x"], obs["y"])
    )

    return env
