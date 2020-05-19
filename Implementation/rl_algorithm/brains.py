from abc import ABC, abstractmethod

import numpy as np
from gym import Space
from gym.spaces import Discrete

from rl_algorithm.parameters import Constant, AlphaVisitDecay, Eligibility
from rl_algorithm.utils import mydefaultdict


class AgentObservation(object):
    def __init__(self, state, action, reward, state2):
        self.state = state
        self.action = action
        self.reward = reward
        self.state2 = state2

    def unpack(self):
        return self.state, self.action, self.reward, self.state2



class Brain(ABC):
    """The class which implements the core of the algorithms"""

    def __init__(self, observation_space: Space, action_space: Space):
        """
        :param observation_space: instance of Space or None. If None, it means that the observation space
                                  is not known a priori and so it is not needed for the algorithm.
        :param action_space:      instance of Space.
        """

        self.observation_space = observation_space
        self.action_space = action_space
        self.policy = None

        self.episode = 0
        self.iteration = 0
        self.episode_iteration = 0
        self.obs_history = []
        self.total_reward = 0

        self.eval = False
        self.last_chosen_action = None

    def set_eval(self, eval: bool):
        self.eval = eval

    def set_policy(self, policy):
        self.policy = policy

    @abstractmethod
    def q_values(self, state, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, state, **kwargs):
        """From a state, return the action for the implemented approach.
        e.g. in Q-Learning, select the argmax of the Q-values relative to the 'state' parameter."""
        raise NotImplementedError

    @abstractmethod
    def observe(self, obs: AgentObservation, *args, **kwargs):
        """Called at each observation.
        E.g. in args there can be the S,A,R,S' tuple for save it in a buffer
        that will be read in the "learn" method."""
        self.obs_history.append(obs)
        self.total_reward += obs.reward

    def update(self, *args, **kwargs):
        """action performed at the end of each iteration
        Subclass implementations should call this method.
        """
        self.episode_iteration += 1
        self.iteration += 1

    def start(self, state):
        self.episode_iteration = 0
        self.total_reward = 0
        self.obs_history = []

    @abstractmethod
    def step(self, obs: AgentObservation, *args, **kwargs):
        """The method performing the learning (e.g. in Q-Learning, update the table)"""
        raise NotImplementedError

    def end(self, obs:AgentObservation, *args, **kwargs):
        """action performed at the end of each episode
        Subclasses implementations should call this method.
        """
        self.episode += 1

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class TDBrain(Brain):

    def __init__(self, observation_space:Discrete, action_space:Discrete, gamma=0.99, alpha=None, lambda_=0):
        super().__init__(observation_space, action_space)

        self.gamma = gamma
        self.alpha = Constant(alpha) if alpha is not None else AlphaVisitDecay(action_space)
        self.lambda_ = lambda_

        self._init()

    def _init(self):
        # sparse representation
        self.Q = mydefaultdict(np.zeros((self.action_space.n,)))
        self.eligibility = Eligibility(self.lambda_, self.gamma)

    def start(self, state):
        super().start(state)
        self.eligibility.reset()
        action = self.choose_action(state)
        return action

    def q_values(self, state, **kwargs):
        Q_values = self.Q[state]  # defaultdict, look at __init__
        return Q_values

    def choose_action(self, state, **kwargs):
        Q_values = self.Q[state]  # defaultdict, look at __init__
        action = self.policy.choose_action(Q_values)
        return action

    def step(self, obs: AgentObservation, *args, **kwargs):
        action2 = self.update_Q(obs)
        self.last_chosen_action = action2
        return action2

    def end(self, obs: AgentObservation, *args, **kwargs):
        super().end(obs)
        if self.eval:
            return
        state, action, reward, state2 = obs.unpack()
        delta = reward - self.Q[state][action]
        for (s, a) in self.eligibility.traces:
            self.Q[s][a] += self.alpha.get(s, a) * delta * self.eligibility.get(s, a)

    @abstractmethod
    def update_Q(self, obs: AgentObservation):
        raise NotImplementedError

    def observe(self, obs:AgentObservation, *args, **kwargs):
        super().observe(obs)
        self.eligibility.to_one(obs.state, obs.action)
        self.alpha.update(obs.state, obs.action)

    def reset(self):
        self._init()


class QLearning(TDBrain):

    def __init__(self, observation_space:Discrete, action_space, gamma=0.99, alpha=None, lambda_=0):
        super().__init__(observation_space, action_space, gamma, alpha, lambda_)

    def update_Q(self, obs:AgentObservation):
        state, action, reward, state2 = obs.unpack()

        # Q-Learning
        action2 = self.choose_action(state2)
        Qa = np.max(self.Q[state2])
        actions_star = np.argwhere(self.Q[state2] == Qa).flatten().tolist()

        delta = reward + self.gamma * Qa - self.Q[state][action]
        for (s, a) in set(self.eligibility.traces.keys()):
            self.Q[s][a] += self.alpha.get(s,a) * delta * self.eligibility.get(s, a)
            # Q-Learning
            if action2 in actions_star:
                self.eligibility.update(s, a)
            else:
                self.eligibility.to_zero(s, a)

        return action2


class Sarsa(TDBrain):

    def __init__(self, observation_space: Discrete, action_space, gamma=0.99, alpha=None, lambda_=0.0):
        super().__init__(observation_space, action_space, gamma, alpha, lambda_)

    def update_Q(self, obs:AgentObservation):
        state, action, reward, state2 = obs.unpack()

        # SARSA
        action2 = self.choose_action(state2)
        Qa = self.Q[state2][action2]

        delta = reward + self.gamma * Qa - self.Q[state][action]
        # if delta!=0.0:
        #     print(len(set(self.eligibility.traces.keys())), delta, reward, Qa, self.Q[state][action])
        for (s, a) in set(self.eligibility.traces.keys()):
            self.Q[s][a] += self.alpha.get(s,a) * delta * self.eligibility.get(s, a)
            # SARSA
            self.eligibility.update(s, a)

        return action2