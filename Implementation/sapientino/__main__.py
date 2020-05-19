#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the main entry-point for the experiments with the Sapientino environment."""
import logging
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import numpy as np
import pythomata
import yaml
from gym.wrappers import Monitor
from gym_sapientino.sapientino_env import SapientinoConfiguration
from pythomata.dfa import DFA

from rl_algorithm.brains import Sarsa, QLearning
from rl_algorithm.callbacks import ModelCheckpoint, plot_history
from rl_algorithm.core import Agent, TrainEpisodeLogger
from rl_algorithm.policies import AutomataPolicy, GreedyQPolicy
from rl_algorithm.utils import learn_dfa, Map
from sapientino.env import make_env_from_dfa, make_env

logging.getLogger("temprl").setLevel(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(level=logging.INFO)
logging.getLogger("rl_algorithm").setLevel(level=logging.INFO)


def parse_args():
    parser = ArgumentParser("sapientino")
    parser.add_argument("--goal-reward", type=int, default=10, help="The reward for satisfying the temporal goal.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="experiments/sapientino-output", help="Output directory for the experiment results.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite the content of the output directory.")
    parser.add_argument("--expert-config", type=str, default="sapientino/expert_config.yaml", help="RL configuration for the expert.")
    parser.add_argument("--learner-config", type=str, default="sapientino/learner_config.yaml", help="RL configuration for the learner.")

    return parser.parse_args()


def run_expert(arguments, configuration):
    agent_dir = Path(arguments.output_dir) / "expert"
    if arguments.overwrite:
        shutil.rmtree(arguments.output_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=False)

    config = SapientinoConfiguration(horizon=50)
    env = make_env(config, arguments.output_dir, arguments.goal_reward)

    np.random.seed(arguments.seed)
    env.seed(arguments.seed)

    policy = AutomataPolicy((-2, ), nb_steps=configuration.nb_exploration_steps, value_max=1.0, value_min=configuration.min_eps)

    agent = Agent(Sarsa(None,
                        env.action_space,
                        gamma=configuration.gamma,
                        alpha=configuration.alpha,
                        lambda_=configuration.lambda_),
                  policy=policy,
                  test_policy=GreedyQPolicy())

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


def run_learner(arguments, configuration, dfa: pythomata.dfa.DFA):
    agent_dir = Path(arguments.output_dir) / "learner"
    shutil.rmtree(agent_dir, ignore_errors=True)
    agent_dir.mkdir(parents=True, exist_ok=False)

    config = SapientinoConfiguration(differential=True, horizon=50)
    env = make_env_from_dfa(config, dfa, goal_reward=arguments.goal_reward)

    np.random.seed(arguments.seed)
    env.seed(arguments.seed)

    policy = AutomataPolicy((-1, ), nb_steps=configuration.nb_exploration_steps, value_max=1.0, value_min=configuration.min_eps)

    algorithm = Sarsa if configuration.algorithm == "sarsa" else QLearning
    agent = Agent(algorithm(None,
                            env.action_space,
                            gamma=configuration.gamma,
                            alpha=configuration.alpha,
                            lambda_=configuration.lambda_),
                  policy=policy,
                  test_policy=GreedyQPolicy())

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
    agent.save(agent_dir / "checkpoints" / "agent.pkl")
    plot_history(history, agent_dir)

    agent = Agent.load(agent_dir / "checkpoints" / "agent.pkl")
    agent.test(Monitor(env, agent_dir / "videos"), nb_episodes=5, visualize=True)

    env.close()


def main(arguments):

    expert_config = Map(yaml.safe_load(open(arguments.expert_config)))
    learner_config = Map(yaml.safe_load(open(arguments.learner_config)))

    print("Run the expert.")
    run_expert(arguments, expert_config)

    print("Learn the automaton from traces.")
    dfa = learn_dfa(arguments)
    dfa_dot_file = os.path.join(arguments.output_dir, "learned_automaton")
    dfa.to_dot(dfa_dot_file)
    print("Check the file {}.svg".format(dfa_dot_file))

    print("Running the learner.")
    run_learner(arguments, learner_config, dfa)


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)

