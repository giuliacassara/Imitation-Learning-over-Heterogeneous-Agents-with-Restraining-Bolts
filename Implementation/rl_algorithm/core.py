import os
import pickle
import time
import timeit
from abc import ABC
from pathlib import Path

from typing import Optional, List

import numpy as np

from rl_algorithm.brains import Brain, AgentObservation
from rl_algorithm.callbacks import Callback, History
from rl_algorithm.policies import Policy, EpsGreedyQPolicy, GreedyQPolicy


class Processor(object):
    """Abstract base class for implementing processors.

    A processor can be necessary if your agent has different requirements with respect to the form of the
    observations, actions, and rewards of the environment. By implementing a custom processor,
    you can effectively translate between the two without having to change the underlying
    implementation of the agent or environment.

    Do not use this abstract base class directly but instead use one of the concrete implementations
    or write your own.
    """

    def process_step(self, observation, reward, done, info):
        """Processes an entire step by applying the processor to the observation, reward, and info arguments.

        # Arguments
            observation (object): An observation as obtained by the environment.
            reward (float): A reward as obtained by the environment.
            done (boolean): `True` if the environment is in a terminal state, `False` otherwise.
            info (dict): The debug info dictionary as obtained by the environment.

        # Returns
            The tuple (observation, reward, done, reward) with with all elements after being processed.
        """
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        return observation, reward, done, info

    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            observation (object): An observation as obtained by the environment

        # Returns
            Observation obtained by the environment processed
        """
        return observation

    def process_reward(self, reward):
        """Processes the reward as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            reward (float): A reward as obtained by the environment

        # Returns
            Reward obtained by the environment processed
        """
        return reward

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.

        # Arguments
            info (dict): An info as obtained by the environment

        # Returns
            Info obtained by the environment processed
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.

        # Arguments
            action (int): Action given to the environment

        # Returns
            Processed action given to the environment
        """
        return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.

        # Arguments
            batch (list): List of states

        # Returns
            Processed list of states
        """
        return batch


class TrainEpisodeLogger(Callback):
    def __init__(self):
        # Some algorithms compute multiple episodes at once since they are multi-threaded.
        # We therefore use a dictionary that is indexed by the episode to separate episodes
        # from each other.
        self.episode_start = {}
        self.observations = {}
        self.rewards = {}
        self.actions = {}
        self.metrics = {}
        self.step = 0
        self.nb_steps = -1

    def on_train_start(self, agent, **kwargs):
        """ Print training values at beginning of training """
        self.train_start = timeit.default_timer()
        self.nb_steps = kwargs['nb_steps']
        print('Training for {} steps ...'.format(self.nb_steps))

    def on_train_end(self, agent, **kwargs):
        """ Print training time at end of training """
        duration = timeit.default_timer() - self.train_start
        print('done, took {:.3f} seconds'.format(duration))

    def on_episode_start(self, agent, **kwargs):
        """ Reset environment variables at beginning of each episode """
        self.episode_start[agent.brain.episode] = timeit.default_timer()
        self.observations[agent.brain.episode] = []
        self.rewards[agent.brain.episode] = []
        self.actions[agent.brain.episode] = []
        self.metrics[agent.brain.episode] = {}

    def on_episode_end(self, agent, **kwargs):
        """ Compute and print training statistics of the episode when done """
        episode = agent.brain.episode
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = self.metrics[episode]

        nb_step_digits = str(int(np.ceil(np.log10(self.nb_steps))) + 1)
        template = '{step: ' + nb_step_digits + 'd}/{nb_steps}: episode: {episode}, duration: {duration: 5.3f}s, episode steps: {episode_steps: 5d}, steps per second: {sps: 5.0f}, episode reward: {episode_reward: 10.3f}, mean reward: {reward_mean: 7.3f} [{reward_min: 10.3f}, {reward_max: 10.3f}], mean action: {action_mean: 5.3f} [{action_min: 5.3f}, {action_max: 5.3f}], mean observation: {obs_mean: 5.3f} [{obs_min: 5.3f}, {obs_max: 5.3f}], {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.nb_steps,
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'obs_mean': np.mean(self.observations[episode]),
            'obs_min': np.min(self.observations[episode]),
            'obs_max': np.max(self.observations[episode]),
            'metrics': ", ".join(["{}: {: 10.3f}".format(metrics_key, np.mean(metrics_values))
                                  for metrics_key, metrics_values in metrics.items()]),
        }
        print(template.format(**variables))

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]

    def on_step_end(self, agent, **kwargs):
        """ Update statistics of episode after each step """
        episode = agent.brain.episode
        self.observations[episode].append(kwargs['observation'])
        self.rewards[episode].append(kwargs['reward'])
        self.actions[episode].append(kwargs['action'])
        for k, v in kwargs['metrics'].items():
            self.metrics[episode].setdefault(k, []).append(v)
        self.step += 1


class Agent(ABC):

    def __init__(self, brain: Brain, processor: Optional[Processor] = None,
                 policy: Optional[Policy] = None, test_policy: Optional[Policy] = None):
        """
        :param brain:               the 'brain', a wrapper for the actual learning algorithm.
        :param processor:           The processor.

        :raises ValueError          if the space of the feature extractor output is different from the space of
                                    the brain input.
        """
        self.brain = brain
        self.processor = Processor() if processor is None else processor
        self.policy = policy if policy is not None else EpsGreedyQPolicy()
        self.test_policy = test_policy if test_policy is not None else GreedyQPolicy()

        self.recent_observation = None

    def set_eval(self, eval: bool):
        """Setter method for "eval" field."""
        self.brain.set_eval(eval)

    def observe(self, state, action, reward, state2, **kwargs):
        """Called at each observation. """
        return self._observe(state, action, reward, state2, **kwargs)

    def _observe(self, features, action, reward, features2, **kwargs):
        obs = AgentObservation(features, action, reward, features2)
        self.brain.observe(obs)
        return obs

    def update(self):
        """Called at the end of each iteration.
        It MUST be called only once for iteration."""
        self.brain.update()

    def start(self, state):
        """Called at the start of each episode.
        It MUST be called only once per episode."""
        return self.brain.start(state)

    def step(self, obs: AgentObservation):
        return self.brain.step(obs)

    def end(self, obs: AgentObservation):
        """Called at the end of each episode.
        It MUST be called only once per episode."""
        self.brain.end(obs)

    def save(self, filepath):
        Path(os.path.dirname(filepath)).mkdir(exist_ok=True, parents=True)
        with open(filepath, "wb") as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(filepath) -> 'Agent':
        with open(filepath, "rb") as fin:
            agent = pickle.load(fin)
        return agent

    def reset(self):
        self.brain.reset()

    def fit(self, env, nb_steps: int = 10000, visualize: bool = False,
            callbacks: List[Callback] = None) -> History:
        """
        Fit the agent.

        :param env: the environment
        :param nb_steps: the number of step of the training.
        :param visualize: visualize the training.
        :param callbacks: list of callbacks to call during training.
        :return:
        """
        callbacks = [] if callbacks is None else callbacks
        history = History()
        callbacks += [history]
        self.policy.set_agent(self)
        self.brain.set_policy(self.policy)

        try:
            for c in callbacks: c.on_train_start(self, nb_steps=nb_steps)
            while self.brain.iteration <= nb_steps:
                self._train_loop(env, render=visualize, callbacks=callbacks)
            for c in callbacks: c.on_train_end(self)
        except KeyboardInterrupt as e:
            pass

        return history

    def _train_loop(self, env, render: bool = False, callbacks: List[Callback] = None):
        agent = self
        callbacks = [] if callbacks is None else callbacks

        state = env.reset()
        self.recent_observation = state
        state = self.processor.process_observation(state)

        done = False

        action = agent.start(state)
        action = self.processor.process_action(action)
        for c in callbacks: c.on_episode_start(self)

        obs = None
        if render:
            env.render(mode='human')

        while not done:
            self.update()
            for c in callbacks: c.on_step_start(self)
            state2, reward, done, info = env.step(action)
            state2, reward, done, info = self.processor.process_step(state2, reward, done, info)
            self.recent_observation = state2
            if render:
                env.render(mode='human')

            obs = agent.observe(state, action, reward, state2)
            for c in callbacks: c.on_step_end(self, observation=state, action=action, reward=reward,
                                              metrics={"q_value": agent.brain.Q[state][action],
                                                       **agent.brain.policy.get_metrics()})

            action = agent.step(obs)
            action = self.processor.process_action(action)
            state = state2

        for c in callbacks: c.on_episode_end(self)
        agent.end(obs)

    def test(self, env, nb_episodes=1, visualize=True):
        print('Testing for {} episodes ...'.format(nb_episodes))

        self.brain.set_eval(True)
        self.test_policy.set_agent(self)
        self.brain.set_policy(self.test_policy)

        try:
            for n in range(nb_episodes):
                total_reward = 0
                steps = 0
                state = env.reset()
                if visualize:
                    time.sleep(0.001)
                    env.render(mode='human')
                done = False

                while not done:
                    action = self.brain.choose_action(state)
                    state2, reward, done, info = env.step(action)
                    total_reward += reward
                    steps += 1
                    if visualize:
                        time.sleep(0.001)
                        env.render(mode='human')

                    state = state2

                template = 'Episode {0}: reward: {1:.3f}, steps: {2}'
                variables = [
                    n + 1,
                    total_reward,
                    steps,
                ]
                print(template.format(*variables))
        except KeyboardInterrupt:
            pass

        self.brain.set_eval(False)
