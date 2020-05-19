#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This is the main entry-point for the experiments with the Breakout environment."""
import logging
import os
from argparse import ArgumentParser

import yaml

from breakout.learner import run_learner
from breakout.expert import run_expert
from rl_algorithm.utils import Map, learn_dfa

logging.getLogger("temprl").setLevel(level=logging.DEBUG)
logging.getLogger("matplotlib").setLevel(level=logging.INFO)
logging.getLogger("rl_algorithm").setLevel(level=logging.INFO)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cols", type=int, default=3, help="Number of columns.")
    parser.add_argument("--rows", type=int, default=3, help="Number of rows.")
    parser.add_argument("--brick-reward", type=int, default=5, help="The reward for breaking a brick.")
    parser.add_argument("--step-reward", type=float, default=-0.01, help="The reward for breaking a brick.")
    parser.add_argument("--goal-reward", type=int, default=1000, help="The reward for satisfying the temporal goal.")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite the content of the output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="experiments/breakout-output", help="Output directory for the experiment results.")
    parser.add_argument("--expert-config", type=str, default="breakout/expert_config.yaml", help="RL configuration for the expert.")
    parser.add_argument("--learner-config", type=str, default="breakout/learner_config.yaml", help="RL configuration for the learner.")
    return parser.parse_args()


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
