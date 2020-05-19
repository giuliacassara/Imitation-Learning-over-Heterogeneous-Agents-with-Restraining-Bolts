import os
from pathlib import Path

import gym
import numpy as np
import pythomata
from flloat.parser.ldlf import LDLfParser
from flloat.semantics import PLInterpretation
from gym.spaces import MultiDiscrete
from gym_minecraft_pygame.minecraft_env import MinecraftConfiguration, item2int, Resources, Tools, Task, ActionSpaceType
from gym_minecraft_pygame.wrappers.dict_space import MinecraftDictSpace
from pythomata.dfa import DFA
from temprl.wrapper import TemporalGoal, TemporalGoalWrapper

from rl_algorithm.temporal import TemporalGoalWrapperLogTraces

int2item = {v: k for k, v in item2int.items()}
LABELS = {i.value for i in item2int.keys() if i}


class MinecraftWrapper(gym.ObservationWrapper):

    def __init__(self, config: MinecraftConfiguration):
        super().__init__(MinecraftDictSpace(config))
        self._previous_completed_tasks = None
        self._completed_tasks = None

    def observation(self, observation):
        new_observation = observation
        new_observation["previous_completed_tasks"] = self._previous_completed_tasks
        self._previous_completed_tasks = np.copy(self._completed_tasks)
        self._completed_tasks = new_observation["completed_tasks"]
        return new_observation

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._previous_completed_tasks = np.copy(obs["completed_tasks"])
        self._completed_tasks = self._previous_completed_tasks
        return obs


class MinecraftExpertWrapper(gym.Wrapper):

    def __init__(self, config: MinecraftConfiguration):
        super().__init__(MinecraftWrapper(config))
        self.observation_space = MultiDiscrete((
            self.unwrapped._x_space.n,
            self.unwrapped._y_space.n,
            self.unwrapped._theta_space.n
        ))

MinecraftLearnerWrapper = MinecraftExpertWrapper


class MinecraftTemporalWrapper(TemporalGoalWrapper):

    def step(self, action):
        obs, reward, done, info = super().step(action)
        some_failed = any(tg.is_failed() for tg in self.temp_goals)
        all_true = all(tg.is_true() for tg in self.temp_goals)
        # print(some_failed, all_true)
        new_done = done or some_failed or all_true
        return obs, reward, new_done, info


def extract_minecraft_fluents(obs, action) -> PLInterpretation:
    item_idx = obs["item"]
    item = int2item[item_idx]
    command = obs["command"]
    previous_completed_tasks = obs["previous_completed_tasks"]
    next_completed_tasks = obs["completed_tasks"]

    fluents = set()

    if command == 1 and isinstance(item, Resources):
        fluents.add(item.value)
    elif command == 2 and isinstance(item, Tools):
        fluents.add(item.value)
    result = PLInterpretation(fluents)
    return result


def make_goal(task: Task) -> str:
    """
    Given a Minecraft Task, define the equivalent goal in LDLf.

    :return: the string associated with the goal.
    """
    task_labels = {a.item.value for a in task.actions}
    empty = "(" + " & ".join("!" + ts for ts in task_labels) + ")"
    f = "<" + empty + "*;{}>tt"
    regexp = (";" + empty + "*;").join([a.item.value for a in task.actions])
    # regexp = regexp + " & done"
    f = f.format(regexp)

    return f


def make_env(config: MinecraftConfiguration, output_dir, goal_reward: float = 1000.0,
             reward_shaping: bool = True) -> gym.Env:
    """
    Make the Minecraft environment.

    :param config: the Minecraft configuration.
    :param output_dir: the path to the output directory.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    temporal_goals = []
    for t in config.tasks:
        formula_string = make_goal(t)
        print("Formula: {}".format(formula_string))
        formula = LDLfParser()(formula_string)
        tg = TemporalGoal(formula=formula,
                          reward=goal_reward,
                          # labels=LABELS,
                          reward_shaping=reward_shaping,
                          zero_terminal_state=False,
                          extract_fluents=extract_minecraft_fluents)
        temporal_goals.append(tg)

        tg._automaton.to_dot(os.path.join(output_dir, "true_automaton_{}".format(t.name)))
        print("Original automaton at {}".format(os.path.join(output_dir, "true_automaton_{}.svg".format(t.name))))

    env = MinecraftTemporalWrapper(
        MinecraftExpertWrapper(config),
        temporal_goals,
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=lambda obs, action:
            (obs["x"], obs["y"], obs["theta"]) if config.action_space_type == ActionSpaceType.DIFFERENTIAL else (obs["x"], obs["y"])
    )

    positive_traces_path = Path(output_dir, "positive_traces.txt")
    negative_traces_path = Path(output_dir, "negative_traces.txt")
    env = TemporalGoalWrapperLogTraces(env, extract_minecraft_fluents, positive_traces_path, negative_traces_path)

    return env


def make_env_from_dfa(config: MinecraftConfiguration, dfa: pythomata.dfa.DFA,
                      goal_reward: float = 1000.0,  reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Minecraft configuration.
    :param dfa: the automaton that constitutes the goal.
    :param goal_reward: the reward associated to the goal.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    tg = TemporalGoal(automaton=dfa,
                      reward=goal_reward,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_minecraft_fluents)

    env = MinecraftTemporalWrapper(
        MinecraftLearnerWrapper(config),
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=lambda obs, action:
            (obs["x"], obs["y"], obs["theta"]) if config.action_space_type == ActionSpaceType.DIFFERENTIAL else (obs["x"], obs["y"])
    )

    return env
