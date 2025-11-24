import datetime as dt
import os
import pickle
import time
from datetime import datetime # for classified the csv files
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter
from gymnasium.envs.classic_control import MountainCarEnv, PendulumEnv
import gymnasium as gym 

import numpy as np
import pygame
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

import matplotlib.pyplot as plt
import cv2

class SGDFunctionApproximator:
    def __init__(self, env):
        # Générez des exemples d'observations en tenant compte du champ de vision limité
        observation_examples = []
        for _ in range(10000):
            obs, _ = env.reset()
            # Simulez une observation partielle (140° devant le fauteuil)
            partial_obs = self._get_partial_obs(obs, env.robot_pos, env.goal_pos, env.objects, env.humans)
            observation_examples.append(partial_obs)
        observation_examples = np.array(observation_examples, dtype='float64')

        # Normalisation
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Utilisez RBF pour la transformation des features
        self.featurizer = pipeline.FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # Initialisez un modèle par dimension d'action
        self.models = []
        for _ in range(env.action_space.shape[0]):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(obs)], [0])
            self.models.append(model)

    def _get_partial_obs(self, full_obs, robot_pos, goal_pos, objects, humans):
        """Return observation limited to 140° in front of the robot."""
        partial_obs = np.concatenate([robot_pos, goal_pos])
        for obj in objects:
            if self._is_in_field_of_view(robot_pos, goal_pos, obj["pos"]):
                partial_obs = np.concatenate([partial_obs, obj["pos"], [obj["radius"]]])
        for human in humans:
            if self._is_in_field_of_view(robot_pos, goal_pos, human["pos"]):
                partial_obs = np.concatenate([partial_obs, human["pos"]])
        return partial_obs

    def _is_in_field_of_view(self, robot_pos, goal_pos, target_pos, angle_deg=140):
        """Check if target is within the field of view (140° in front of the robot)."""
        direction = goal_pos - robot_pos
        target_dir = target_pos - robot_pos
        angle = np.degrees(np.arctan2(target_dir[1], target_dir[0]) - np.arctan2(direction[1], direction[0]))
        return abs(angle) <= angle_deg / 2

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]
    
    def predict(self, state, action_index=None):
        """Predict Q-value for a state (and optional action index)."""
        features = self.featurize_state(state)
        if action_index is None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[action_index].predict([features])[0]

    def update(self, state, action_index, td_target):
        """Update the model for a given state and action index."""
        features = self.featurize_state(state)
        self.models[action_index].partial_fit([features], [td_target])


class Tamer:
    """
    TAMER agent for continuous action space, with automatic human feedback based on social zones.
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0,
        min_eps=0,
        tame=True,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None,
    ):
        self.env = env
        self.tame = tame
        self.ts_len = ts_len
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Initialize model
        if model_file_to_load is not None:
            self.load_model(model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env)  # Human feedback model

        # Reward logging
        self.reward_log_columns = [
            'Episode', 'Timestep', 'Human Feedback', 'Environment Reward', 'Total Reward'
        ]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reward_log_path = os.path.join(self.output_dir, f'tamer_rewards_log_{current_time}.csv')

    def act(self, state):
        """Epsilon-greedy policy for continuous action space."""
        if np.random.random() < 1 - self.epsilon:
            return self.H.predict(state)
        else:
            return np.random.uniform(-1, 1, size=self.env.action_space.shape)

    def _get_human_feedback(self, state):
        """Calculate automatic feedback from humans based on social zones."""
        feedback = 0
        robot_pos = state[:2]
        for human in self.env.humans:
            dist = np.linalg.norm(robot_pos - human["pos"])
            if dist < 0.4:  # Intimate zone
                feedback -= 5.0
            elif dist < 1.2:  # Personal zone
                feedback -= 2.0
        return feedback

    def _train_episode(self, episode_index):
        """Train for one episode with automatic human feedback."""
        state, _ = self.env.reset()
        total_reward = 0
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            if episode_index == 0:
                dict_writer.writeheader()
            for ts in count():
                action = self.act(state)
                next_state, env_reward, done, _, _ = self.env.step(action)
                human_feedback = self._get_human_feedback(state)
                total_reward += env_reward

                if self.tame:
                    # Update model with human feedback
                    for i in range(len(action)):
                        self.H.update(state, i, human_feedback)

                # Log rewards
                dict_writer.writerow({
                    'Episode': episode_index + 1,
                    'Timestep': ts,
                    'Human Feedback': human_feedback,
                    'Environment Reward': env_reward,
                    'Total Reward': total_reward
                })

                if done:
                    print(f'Episode: {episode_index + 1}, Total Reward: {total_reward}')
                    break

                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    def train(self, model_file_to_save=None):
        """Run the training loop for all episodes."""
        for i in range(self.num_episodes):
            self._train_episode(i)
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(model_file_to_save)

    def play(self, n_episodes=1, render=False):
        """Run episodes with the trained agent."""
        self.epsilon = 0
        rewards = []
        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                if render:
                    self.env.render()
                state = next_state
            rewards.append(total_reward)
            print(f'Episode: {i + 1}, Reward: {total_reward}')
        self.env.close()
        return rewards

    def evaluate(self, n_episodes=10):
        """Evaluate the agent over a number of episodes."""
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return avg_reward

    def save_model(self, filename):
        """Save the model to disk."""
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(self.H, f)

    def load_model(self, filename):
        """Load a model from disk."""
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H = pickle.load(f)