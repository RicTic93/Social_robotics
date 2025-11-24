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

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
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

    def _get_partial_obs(self, obs, robot_pos, goal_pos, objects, humans):
        # Filtrez les objets et humains dans le champ de vision de 140° devant le fauteuil
        # (à implémenter selon votre logique de détection angulaire)
        pass

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None,  # filename of pretrained model
        feedback_mode='auto',  # 'keyboard', 'mouse', or 'auto'
        action_map= MOUNTAINCAR_ACTION_MAP  # action map for interface
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.feedback_mode = feedback_mode
        self.action_map = action_map
        
        '''# Automatically determine feedback mode if set to 'auto'
        if self.feedback_mode == 'auto':
            if hasattr(env,'env') and isinstance(env.env, PendulumEnv):
                self.feedback_mode = 'mouse'
                self.action_map = {0: 'left', 1: 'neutral', 2: 'right'}
            elif isinstance(self.env.unwrapped, gym.envs.classic_control.PendulumEnv):
                self.feedback_mode = 'keyboard'
                self.action_map = MOUNTAINCAR_ACTION_MAP  # not needed for mouse feedback
            else:
                self.feedback_mode = feedback_mode # default to whatever was passed
                self.action_map = action_map
            # print Environment, feedback mode and action map
            print(f"Environment detected as : {type(env.unwrapped)}")
            print(f"Feedback mode set to: {self.feedback_mode}")
            print(f"Action map set to: {self.action_map}")'''

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        # I modified here to classify the csv files by time
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reward_log_path = os.path.join(self.output_dir, f'tamer_rewards_log_{current_time}.csv')

        #self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)
    
    def _get_human_feedback(self, state):
        # Calculez le feedback des humains en fonction de la distance
        feedback = 0
        robot_pos = state[:2]
        for human in self.env.humans:
            dist = np.linalg.norm(robot_pos - human["pos"])
            if dist < 0.4:  # Zone intime
                feedback -= 5.0
            elif dist < 1.2:  # Zone personnelle
                feedback -= 2.0
        return feedback

    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}  Timestep:', end='')
        # print(f'Episode: {episode_index + 1}  Timestep:')
        cv2.namedWindow('OpenAI Gymnasium Training', cv2.WINDOW_NORMAL)

        rng = np.random.default_rng()
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                print(f' {ts}', end='')
                # print(f' {ts}')
                # self.env.render()
                frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                # print(self.env.render())
                cv2.imshow('OpenAI Gymnasium Training', frame_bgr)
                key = cv2.waitKey(25)  # Adjust the delay (25 milliseconds in this case)
                if key == 27:
                    break

                # Determine next action
                action = self.act(state)
                if self.tame:
                    disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info, _ = self.env.step(action)

                if not self.tame:
                    if done and next_state[0] >= 0.5:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        frame = None

                        time.sleep(0.01)  # save the CPU

                        human_reward = disp.get_feedback()
                        feedback_ts = dt.datetime.now().time()
                        if human_reward != 0:
                            dict_writer.writerow(
                                {
                                    'Episode': episode_index + 1,
                                    'Ep start ts': ep_start_time,
                                    'Feedback ts': feedback_ts,
                                    'Human Reward': human_reward,
                                    'Environment Reward': reward
                                }
                            )
                            self.H.update(state, action, human_reward)
                            break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    cv2.destroyAllWindows()
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    async def train(self, model_file_to_save=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        # render first so that pygame display shows up on top
        # self.env.render()
        
        disp = None
        if self.tame:
            # only init pygame display if we're actually training tamer
            from .interface import Interface
            disp = Interface(action_map=self.action_map, feedback_mode=self.feedback_mode)

        for i in range(self.num_episodes):
            print(f"Num episode : {i}")
            self._train_episode(i, disp)

        print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(filename=model_file_to_save)
        #if self.tame and disp is not None:
            #pygame.quit()  # Clean up Pygame after training

    def play(self, n_episodes=1, render=False):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        if render:            
            cv2.namedWindow('OpenAI Gymnasium Playing', cv2.WINDOW_NORMAL)

        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()[0]
            done = False
            tot_reward = 0
            # TODO setup a duration criterion in case of impossibility to find a solution
            while not done:
                action = self.act(state)
                next_state, reward, done, info, _ = self.env.step(action)
                tot_reward += reward
                if render:
                    # self.env.render()
                    frame_bgr = cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)
                    cv2.imshow('OpenAI Gymnasium Playing', frame_bgr)
                    key = cv2.waitKey(25)  # Adjust the delay (25 milliseconds in this case)
                    if key == 27:
                        break
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        self.env.close()
        if render:
            cv2.destroyAllWindows()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model
