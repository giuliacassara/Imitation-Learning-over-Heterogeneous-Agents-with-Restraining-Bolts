import os
import shutil
from pathlib import Path

import gym
import numpy as np
from flloat.parser.ldlf import LDLfParser
from gym.spaces import MultiDiscrete
from gym.wrappers import Monitor
from gym_breakout_pygame.breakout_env import BreakoutConfiguration
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal

from breakout.common import BreakoutWrapper, extract_breakout_fluents, make_goal
from rl_algorithm.brains import Sarsa, QLearning
from rl_algorithm.callbacks import ModelCheckpoint, plot_history
from rl_algorithm.core import TrainEpisodeLogger, Agent
from rl_algorithm.policies import EpsGreedyQPolicy, AutomataPolicy
from rl_algorithm.temporal import TemporalGoalWrapperLogTraces


class BreakoutExpertWrapper(gym.Wrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(BreakoutWrapper(*args, **kwargs))

        self.observation_space = MultiDiscrete((
            self.env.observation_space.spaces["paddle_x"].n,
        ))


def make_env(config: BreakoutConfiguration, output_dir, goal_reward: float = 1000.0,
             reward_shaping: bool = True) -> gym.Env:
    """
    Make the Breakout environment.

    :param config: the Breakout configuration.
    :param output_dir: the path to the output directory.
    :param reward_shaping: apply automata-based reward shaping.
    :return: the Gym environment.
    """
    unwrapped_env = BreakoutExpertWrapper(config)

    formula_string = make_goal(config.brick_cols)
    formula = LDLfParser()(formula_string)
    labels = {"c{}".format(i) for i in range(config.brick_cols)}
    tg = TemporalGoal(formula=formula,
                      reward=goal_reward,
                      labels=labels,
                      reward_shaping=reward_shaping,
                      zero_terminal_state=False,
                      extract_fluents=extract_breakout_fluents)

    print("Formula: {}".format(formula_string))
    tg._automaton.to_dot(os.path.join(output_dir, "true_automaton"))
    print("Original automaton at {}".format(os.path.join(output_dir, "true_automaton.svg")))

    env = TemporalGoalWrapper(
        unwrapped_env,
        [tg],
        combine=lambda obs, qs: tuple((*obs, *qs)),
        feature_extractor=(lambda obs, action: (
            obs["paddle_x"],
        ))
    )

    positive_traces_path = Path(output_dir, "positive_traces.txt")
    negative_traces_path = Path(output_dir, "negative_traces.txt")
    env = TemporalGoalWrapperLogTraces(env, extract_breakout_fluents, positive_traces_path, negative_traces_path)

    return env


def run_expert(arguments, configuration):
    agent_dir = Path(arguments.output_dir) / "expert"
    if arguments.overwrite:
        shutil.rmtree(arguments.output_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=False)

    config = BreakoutConfiguration(brick_rows=arguments.rows, brick_cols=arguments.cols,
                                   brick_reward=arguments.brick_reward, step_reward=arguments.step_reward,
                                   ball_enabled=False, fire_enabled=True)
    env = make_env(config, arguments.output_dir, arguments.goal_reward)

    np.random.seed(arguments.seed)
    env.seed(arguments.seed)

    policy = AutomataPolicy((-2, ), nb_steps=configuration.nb_exploration_steps, value_max=1.0, value_min=configuration.min_eps)

    algorithm = Sarsa if configuration.algorithm == "sarsa" else QLearning
    agent = Agent(algorithm(None,
                        env.action_space,
                        gamma=configuration.gamma,
                        alpha=configuration.alpha,
                        lambda_=configuration.lambda_),
                  policy=policy,
                  test_policy=EpsGreedyQPolicy(eps=0.01))

    history = agent.fit(
        env,
        nb_steps=configuration.nb_steps,
        visualize=configuration.visualize_training,
        callbacks=[
            ModelCheckpoint(str(agent_dir / "checkpoints" / "agent-{}.pkl")),
            TrainEpisodeLogger()
        ]
    )
    history.save(agent_dir / "history.json")
    agent.save(Path(agent_dir, "checkpoints", "agent.pkl"))
    plot_history(history, agent_dir)

    agent = Agent.load(agent_dir / "checkpoints" / "agent.pkl")
    agent.test(Monitor(env, agent_dir / "videos"), nb_episodes=5, visualize=True)

    env.close()
