import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from rl_algorithm.brains import AgentObservation


class Callback(object):

    def on_train_start(self, agent, **kwargs):
        pass

    def on_train_end(self, agent, **kwargs):
        pass

    def on_test_start(self, agent, **kwargs):
        pass

    def on_test_end(self, agent, **kwargs):
        pass

    def on_episode_start(self, agent, **kwargs):
        pass

    def on_episode_end(self, agent, **kwargs):
        pass

    def on_step_start(self, agent, **kwargs):
        pass

    def on_step_end(self, agent, **kwargs):
        pass


class ModelCheckpoint(Callback):
    """Save the model after every episode."""

    def __init__(self, filepath: str, period: int = 100):
        self.filepath = filepath
        self.period = period
        self.episode_since_last_save = -1

    def on_episode_end(self, agent, **kwargs):
        self.episode_since_last_save += 1
        if self.episode_since_last_save >= self.period:
            self.episode_since_last_save = 0
            agent.save(self.filepath.format(agent.brain.episode))


class EpisodeStats:

    def __init__(self):
        self.rewards = np.array([], dtype=np.float64)
        self.actions = np.array([], dtype=np.float64)
        self.observations = np.array([], dtype=np.float64)
        self.q_values = np.array([], dtype=np.float64)

    def append(self, obs: AgentObservation, q_value: float = np.nan):
        self.rewards = np.append(self.rewards, obs.reward)
        self.actions = np.append(self.actions, obs.action)
        self.observations = np.append(self.observations, obs.state)
        self.q_values = np.append(self.q_values, q_value)

    @property
    def json(self):
        return {
            "rewards": list(self.rewards),
            "actions": list(self.actions),
            "observations": list(self.observations),
            "q_values": list(self.q_values)
        }

    @classmethod
    def from_json(cls, d: Dict):
        e = EpisodeStats()
        e.rewards = np.asarray(d["rewards"])
        e.observations = np.asarray(d["observations"])
        e.actions = np.asarray(d["actions"])
        e.q_values = np.asarray(d["q_values"])
        return e


class History(Callback):
    """Track the history of the training."""

    def __init__(self):
        super().__init__()
        self.episodes = []  # type: List[EpisodeStats]

    def on_episode_start(self, agent, **kwargs):
        self.episodes.append(EpisodeStats())

    def on_step_end(self, agent, **kwargs):
        last_obs = agent.brain.obs_history[-1]
        last_q_value = agent.brain.Q[last_obs.state][agent.brain.last_chosen_action]
        self.episodes[agent.brain.episode].append(last_obs, last_q_value)

    def save(self, filepath):
        json_episodes = list(map(lambda e: e.json, self.episodes))
        with open(filepath, "w") as fout:
            json.dump(json_episodes, fout)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "r") as fin:
            json_episodes = json.load(fin)
        h = History()
        h.episodes = [EpisodeStats.from_json(e) for e in json_episodes]
        return h


def plot_history(history: History, directory: str):
    plt.plot(np.asarray([np.mean(e.rewards) for e in history.episodes]))

    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.grid(True)
    plt.savefig(os.path.join(directory, "avg-reward.png"))
    plt.clf()

    plt.plot(np.asarray([np.mean(e.q_values) for e in history.episodes]))
    plt.xlabel('Episodes')
    plt.ylabel('Average Q-value')
    plt.grid(True)
    plt.savefig(os.path.join(directory, "avg-q-values.png"))
    plt.clf()
