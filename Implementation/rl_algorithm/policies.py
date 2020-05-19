import logging
from collections import defaultdict
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


class Policy(object):
    """Abstract base class for all implemented policies.

    Each policy helps with selection of action to take on an environment.

    Do not use this abstract base class directly but instead use one of the concrete policies implemented.
    To implement your own policy, you have to implement the following methods:

    - `select_action`

    # Arguments
        agent (rl.core.Agent): Agent used
    """

    def set_agent(self, agent):
        self.agent = agent

    def choose_action(self, q_values, **kwargs):
        raise NotImplementedError()

    def get_metrics(self):
        return {}


class LinearAnnealedPolicy(Policy):
    """Implement the linear annealing policy

    Linear Annealing Policy computes a current threshold value and
    transfers it to an inner policy which chooses the action. The threshold
    value is following a linear function decreasing over time."""

    def __init__(self, inner_policy: Policy, attr, value_max, value_min, value_test, nb_steps):
        if not hasattr(inner_policy, attr):
            raise ValueError('Policy does not have attribute "{}".'.format(attr))

        super(LinearAnnealedPolicy, self).__init__()

        self.inner_policy = inner_policy
        self.attr = attr
        self.value_max = value_max
        self.value_min = value_min
        self.value_test = value_test
        self.nb_steps = nb_steps

    def get_current_value(self):
        """Return current annealing value

        # Returns
            Value to use in annealing
        """
        if not self.agent.brain.eval:
            # Linear annealed: f(x) = ax + b.
            a = -float(self.value_max - self.value_min) / float(self.nb_steps)
            b = float(self.value_max)
            value = max(self.value_min, a * float(self.agent.brain.iteration) + b)
        else:
            value = self.value_test
        return value

    def choose_action(self, q_values, **kwargs):
        """Choose an action to perform

        # Returns
            Action to take (int)
        """
        setattr(self.inner_policy, self.attr, self.get_current_value())
        return self.inner_policy.choose_action(q_values, **kwargs)

    def get_metrics(self):
        return self.inner_policy.get_metrics()


class EpsGreedyQPolicy(Policy):
    """Implement the epsilon greedy policy

    Eps Greedy policy either:

    - takes a random action with probability epsilon
    - takes current best action with prob (1 - epsilon)
    """

    def __init__(self, eps=.1):
        super(EpsGreedyQPolicy, self).__init__()
        self.eps = eps

    def choose_action(self, q_values, **kwargs):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        nb_actions = q_values.shape[0]

        if np.random.uniform() < self.eps:
            action = np.random.random_integers(0, nb_actions - 1)
        else:
            action = np.argmax(q_values)
        return action

    def get_metrics(self):
        return {"eps": self.eps}


class GreedyQPolicy(Policy):
    """Implement the greedy policy

    Greedy policy returns the current best action according to q_values
    """

    def choose_action(self, q_values, **kwargs):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        assert q_values.ndim == 1
        action = np.argmax(q_values)
        return action


class AutomataPolicy(Policy):

    def __init__(self, automata_states_indexes: Tuple[int] = (-1, ), nb_steps=50000, value_max=1.0, value_min=.01):

        self.automata_states_indexes = automata_states_indexes
        self.nb_steps = nb_steps
        self.value_max = value_max
        self.value_min = value_min

        self.policies = defaultdict(self.policy_factory)
        self.counts = defaultdict(int)

    def extract_q(self, obs):
        return tuple(obs[index] for index in self.automata_states_indexes)

    def policy_factory(self):
        policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=self.value_max,
                                      value_min=self.value_min, value_test=.0, nb_steps=self.nb_steps)
        policy.set_agent(self.agent)
        return policy

    def set_agent(self, agent):
        super().set_agent(agent)
        for p in self.policies.values():
            p.set_agent(agent)

    def choose_action(self, q_values, **kwargs):
        """Return the selected action

        # Arguments
            q_values (np.ndarray): List of the estimations of Q for each action

        # Returns
            Selection action
        """
        q = ()
        if self.agent.recent_observation is not None:
            obs = self.agent.recent_observation
            q = self.extract_q(obs)
        self.counts[q] += 1

        selected_policy = self.policies[q]

        old_step = self.agent.brain.iteration
        self.agent.brain.iteration = self.counts[q]
        sampled_action = selected_policy.choose_action(q_values, **kwargs)
        self.agent.brain.iteration = old_step

        return sampled_action

    def get_metrics(self):
        logger.debug("epsilon values: {}"
                     .format({"eps-{}".format(k): p.inner_policy.eps for k, p in self.policies.items()}))
        return {"mean-eps": np.mean([p.inner_policy.eps for k, p in self.policies.items()])}
