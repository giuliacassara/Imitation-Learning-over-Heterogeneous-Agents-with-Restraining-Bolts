# -*- coding: utf-8 -*-
"""This package contains utilities for the temporal goal gym wrapper."""
import logging
from copy import deepcopy
from typing import Optional, List

import gym
from flloat.semantics import PLInterpretation
from pythomata.base import State
from pythomata.dfa import DFA
from temprl.wrapper import TemporalGoalWrapper, TemporalGoal

logger = logging.getLogger(__name__)
EMPTY = "_"
SEPARATOR = ";"


class TemporalGoalLogger(TemporalGoal):
    """This class represents a dummy temporal goal for keeping track of the traces.."""

    def __init__(self, extract_fluents):
        """
        Initialize the fluents logger.

        :param extract_fluents: the fluents extractor.
        """
        if extract_fluents is not None:
            setattr(self, "extract_fluents", extract_fluents)

        self.cur_trace = []  # type: List[PLInterpretation]
        self.traces = []  # type: List[List[PLInterpretation]]

    def extract_fluents(self, obs, action) -> PLInterpretation:
        """
        Extract high-level features from the observation.

        :return: the list of active fluents.
        """
        raise NotImplementedError

    def step(self, observation, action) -> Optional[State]:
        """Do a step in the simulation."""
        fluents = self.extract_fluents(observation, action)
        self.cur_trace.append(fluents)
        return 0

    def reset(self):
        """Reset the simulation."""
        self.traces.append(self.cur_trace)
        self.cur_trace = []
        return 0

    def observe_reward(self, **kwargs) -> float:
        """Observe the reward of the last transition."""
        return 0.0

    def is_true(self):
        """Check if the simulation is in a final state."""
        return True

    def is_failed(self):
        """Check whether the simulation has failed."""
        return False


class TemporalGoalWrapperLogTraces(gym.Wrapper):

    def __init__(self, env: TemporalGoalWrapper,
                 extract_fluents,
                 positive_traces_filepath: str = "",
                 negative_traces_filepath: str = ""):
        """

        :param env:
        :param positive_traces_filepath: the file where to save the
        :param negative_traces_filepath:
        :param kwargs:
        """
        super().__init__(env)
        self.positive_traces_output_file = open(positive_traces_filepath, "w")
        self.negative_traces_output_file = open(negative_traces_filepath, "w")
        self.positive_traces = []
        self.negative_traces = []

        self.logger = TemporalGoalLogger(extract_fluents)
        env.temp_goals.append(self.logger)

    def reset(self, **kwargs):
        """Reset the Gym environment."""
        trace = self.logger.cur_trace
        temp_goal_all_true = all(tg.is_true() for tg in self.temp_goals)
        if temp_goal_all_true:
            self.positive_traces.append(trace)
        else:
            self.negative_traces.append(trace)

        return super().reset()

    def close(self):

        def compute_trace_string(trace):
            trace_string = SEPARATOR.join("_".join(sorted(prop_int.true_propositions)) for prop_int in trace
                                          if len(prop_int.true_propositions) != 0)
            return trace_string

        for t in self.positive_traces:
            self.positive_traces_output_file.write(compute_trace_string(t) + "\n")
        for t in self.negative_traces:
            self.negative_traces_output_file.write(compute_trace_string(t) + "\n")
        return super().close()
