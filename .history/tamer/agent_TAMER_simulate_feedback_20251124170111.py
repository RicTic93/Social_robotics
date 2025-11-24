import numpy as np
import os
import pickle
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from csv import DictWriter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import math

# Chemins pour sauvegarder les modèles et logs
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

# Crée les dossiers s'ils n'existent pas
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

class SGDFunctionApproximator:
    """Approximation de fonction avec RBF et SGD pour un espace d'action continu."""

    def __init__(self, env, max_objects_in_view=10, max_humans_in_view=10):
        self.max_objects_in_view = max_objects_in_view
        self.max_humans_in_view = max_humans_in_view

        # Génère des exemples d'observations partielles
        observation_examples = []
        for _ in range(10000):
            obs, _ = env.reset()
            observation_examples.append(obs)  # obs est déjà une observation partielle de taille 54

        observation_examples = np.array(observation_examples, dtype='float64')

        # Normalisation des observations
        self.scaler = StandardScaler()
        self.scaler.fit(observation_examples)

        # Transformation RBF
        self.featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        sample_feat = self.featurize_state(observation_examples[0]).shape[0]
        
        # Création d’un SGDRegressor par dimension d’action
        self.models = []
        for _ in range(env.action_space.shape[0]):
            model = SGDRegressor(
                loss="squared_error",
                learning_rate="constant",
                eta0=0.01,
                max_iter=1,
                tol=None,
                random_state=42
            )
            # initialisation obligatoire pour partial_fit
            model.partial_fit([np.zeros(sample_feat)], [0.0])
            self.models.append(model)

    def featurize_state(self, state):
        """Transforme l'état en features pour l'apprentissage."""
        assert np.all(np.isfinite(state)), f"State contains non-finite values: {state}"
        scaled = self.scaler.transform([state])
        features = self.featurizer.transform(scaled)[0]
        assert np.all(np.isfinite(scaled)), f"Scaled state contains non-finite values: {scaled}"
        return features

    def predict(self, state, action_index=None):
        """Prédit la valeur Q pour un état (et optionnellement une action)."""
        features = self.featurize_state(state)
        if action_index is None:
            action = np.array([m.predict([features])[0] for m in self.models])
            #print(f"Predicted Q-values: {action}")
            return action
        else:
            return self.models[action_index].predict([features])[0]

    def update(self, state, action_index, td_target):
        """Met à jour le modèle pour une action donnée."""
        features = self.featurize_state(state)
        self.models[action_index].partial_fit([features], [td_target])

class Tamer:
    """Agent TAMER pour un espace d'action continu avec feedback automatique."""

    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,
        epsilon=0.2,
        min_eps=0.01,
        tame=True,
        ts_len=0.2,
        output_dir=LOGS_DIR,
        model_file_to_load=None,
        max_objects_in_view=10,
        max_humans_in_view=10
    ):
        self.env = env
        self.tame = tame
        self.ts_len = ts_len
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0.5
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # Initialise le modèle
        if model_file_to_load is not None:
            self.load_model(model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env, max_objects_in_view, max_humans_in_view)

        # Configuration du logging
        self.reward_log_columns = [
            'Episode', 'Timestep', 'Human Feedback', 'Environment Reward', 'Total Reward'
        ]
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reward_log_path = os.path.join(self.output_dir, f'tamer_rewards_log_{current_time}.csv')

    def act(self, state):
        """Politique epsilon-greedy pour un espace d'action continu."""
        assert len(state) == 54, f"State size is {len(state)}, expected 54"
        if np.random.random() < 1 - self.epsilon:
            action = np.random.uniform(-1, 1, size=self.env.action_space.shape)
            print(f"Random Action: {action}")
        else:
            action = self.H.predict(state)
            print(f"Predicted Action: {action}")
            
        # Normalise et amplifie l'action pour s'assurer qu'elle a un effet visible
        action_norm = np.linalg.norm(action)
        if action_norm < 0.1:  # Si l'action est trop petite
            action = action / max(action_norm, 1e-7) * 0.5  # Normalise et amplifie

        return action

    def _get_human_feedback(self, state):
        feedback = 0.0
        robot_pos = np.array(state[:2])
        goal_pos = np.array(state[2:4])
        robot_fov_distance = getattr(self.env,"fov_distance", ())
        
        # --- Feedback humain visibles ---
        for human in self.env.humans:
            human_pos = np.array(human["pos"])
            dist = np.linalg.norm(robot_pos - human["pos"])
            if self.env._is_in_field_of_view(robot_pos, goal_pos, human_pos):
            # Vérifie si le robot est dans le champ de vision de l'humain (180°)
                if self.env._is_robot_in_human_fov(human["pos"], human["direction"], robot_pos):
                    if dist < 0.4:  # Zone intime
                        feedback -= 5.0
                    elif dist < 1.2:  # Zone personnelle
                        feedback -= 2.0
                    elif dist < 2.0:  # Zone sociale
                        feedback += 2.0
                    else:
                        feedback += 2.0
                
        # --- Feedback obstacles visibles ---
        # Supposons que env.obstacles est une liste de positions [(x, y), ...]
        for obs in self.env.objects:
            obs_pos = np.array(obj["pos"])
            obs_radius = obj["radius"]
            dist = np.linalg.norm(robot_pos - obs_pos) - obs_radius
            if self.env._is_in_field_of_view(robot_pos, goal_pos, human_pos):
                if dist_obs < 0.3:       # trop proche -> collision probable
                    feedback -= 5.0
                elif dist < 0.6 :

        # Feedback limites/murs
        x, y = robot_pos
        if not (0 <= x <= self.env.width and 0 <= y <= self.env.height):
            feedback -= 5.0        
                
        return feedback



    def _train_episode(self, episode_index):
        """Entraîne l'agent pour un épisode."""
        state, _ = self.env.reset()
        total_reward = 0
        
        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            if episode_index == 0:
                dict_writer.writeheader()

            for ts in count():
                action = self.act(state)
                next_state, env_reward, done, _, info = self.env.step(action)
                
                # Feedback automatique combiné
                human_feedback = self._get_human_feedback(next_state)
                total_reward += env_reward

                if self.tame:
                    # Met à jour le modèle avec le feedback humain
                    for i in range(len(action)):
                        self.H.update(state, i, human_feedback)

                # Log les récompenses
                dict_writer.writerow({
                    'Episode': episode_index + 1,
                    'Timestep': ts,
                    'Human Feedback': human_feedback,
                    'Environment Reward': env_reward,
                    'Total Reward': total_reward
                })

                if done:
                    #print(f'Episode: {episode_index + 1}, Total Reward: {total_reward}')
                    break

                state = next_state

        # Décroissance d'epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    def train(self, model_file_to_save=None):
        """Boucle d'entraînement pour tous les épisodes."""
        for i in range(self.num_episodes):
            self._train_episode(i)
        self.env.close()
        if model_file_to_save is not None:
            self.save_model(model_file_to_save)

    def play(self, n_episodes=1, render=False):
        """Exécute des épisodes avec l'agent entraîné."""
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
                    # Ajoute une pause pour voir l'animation
                    pygame.time.delay(50)  # Pause de 50 ms
                    # Gestion des événements Pygame pour éviter le gel
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            return rewards
                state = next_state
            rewards.append(total_reward)
            print(f'Episode: {i + 1}, Reward: {total_reward}')
        self.env.close()
        return rewards

    def evaluate(self, n_episodes=10):
        """Évalue l'agent sur un nombre d'épisodes."""
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(f'Average total episode reward over {n_episodes} episodes: {avg_reward:.2f}')
        return avg_reward

    def save_model(self, filename):
        """Sauvegarde le modèle sur le disque."""
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(self.H, f)

    def load_model(self, filename):
        """Charge un modèle depuis le disque."""
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            self.H = pickle.load(f)
