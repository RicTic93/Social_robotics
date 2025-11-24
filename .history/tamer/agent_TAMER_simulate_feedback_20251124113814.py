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
            assert len(obs) == 54, f"Observation size is {len(obs)}, expected 54"
            observation_examples.append(obs)
        '''partial_obs = self._get_partial_obs(
            obs, env.robot_pos, env.goal_pos, env.objects, env.humans
        )
        observation_examples.append(partial_obs)'''

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

        # Initialise un modèle par dimension d'action
        self.models = []
        for _ in range(env.action_space.shape[0]):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(observation_examples[0])], [0])
            self.models.append(model)

    def _get_partial_obs(self, full_obs, robot_pos, goal_pos, objects, humans):
        """Retourne une observation partielle (140° devant le fauteuil) de taille fixe."""
        # Taille totale : robot_pos (2) + goal_pos (2) + objets (3*max) + humains (2*max)
        total_size = 4 + 3 * self.max_objects_in_view + 2 * self.max_humans_in_view
        partial_obs = np.zeros(total_size, dtype='float64')

        # Remplit robot_pos et goal_pos
        partial_obs[:2] = robot_pos
        partial_obs[2:4] = goal_pos

        # Ajoute les objets visibles
        obj_start = 4
        obj_count = 0
        for obj in objects:
            if self._is_in_field_of_view(robot_pos, goal_pos, obj["pos"]):
                if obj_count < self.max_objects_in_view:
                    partial_obs[obj_start + 3*obj_count : obj_start + 3*obj_count + 2] = obj["pos"]
                    partial_obs[obj_start + 3*obj_count + 2] = obj["radius"]
                    obj_count += 1

        # Ajoute les humains visibles
        human_start = 4 + 3 * self.max_objects_in_view
        human_count = 0
        for human in humans:
            if self._is_in_field_of_view(robot_pos, goal_pos, human["pos"]):
                if human_count < self.max_humans_in_view:
                    partial_obs[human_start + 2*human_count : human_start + 2*human_count + 2] = human["pos"]
                    human_count += 1

        return partial_obs

    def _is_in_field_of_view(self, robot_pos, goal_pos, target_pos, angle_deg=140):
    """
    Vérifie si target_pos est dans le champ de vision (140°) devant le fauteuil.
    :param robot_pos: Position du robot (tableau de taille 2).
    :param goal_pos: Position du but (tableau de taille 2).
    :param target_pos: Position de la cible (objet ou humain).
    :param angle_deg: Angle du champ de vision (140° par défaut).
    :return: True si la cible est dans le champ de vision.
    """
    # Calcule la direction du robot vers le but
    direction = goal_pos - robot_pos
    direction_angle = math.atan2(direction[1], direction[0])

    # Calcule la direction du robot vers la cible
    target_dir = target_pos - robot_pos
    target_angle = math.atan2(target_dir[1], target_dir[0])

    # Calcule la différence d'angle (en radians)
    angle_diff = target_angle - direction_angle

    # Normalise l'angle entre -π et π
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    # Convertit en degrés et vérifie si dans le champ de vision
    return abs(math.degrees(angle_diff)) <= angle_deg / 2


    def featurize_state(self, state):
        """Transforme l'état en features pour l'apprentissage."""
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]

    def predict(self, state, action_index=None):
        """Prédit la valeur Q pour un état (et optionnellement une action)."""
        features = self.featurize_state(state)
        if action_index is None:
            return np.array([m.predict([features])[0] for m in self.models])
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
        epsilon=0,
        min_eps=0,
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
        self.epsilon = epsilon if not tame else 0
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
        assert len(state) == 54, f"State size is {len(state)}, expected 54"
        if np.random.random() < 1 - self.epsilon:
            return self.H.predict(state)
        else:
            return np.random.uniform(-1, 1, size=self.env.action_space.shape)

    def _get_human_feedback(self, state):
        """Calcule le feedback automatique basé sur les zones sociales."""
        feedback = 0
        robot_pos = state[:2]
        for human in self.env.humans:
            dist = np.linalg.norm(robot_pos - human["pos"])
            if dist < 0.4:  # Zone intime
                feedback -= 5.0
            elif dist < 1.2:  # Zone personnelle
                feedback -= 2.0
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
                next_state, env_reward, done, _, _ = self.env.step(action)
                human_feedback = self._get_human_feedback(state)
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
                    print(f'Episode: {episode_index + 1}, Total Reward: {total_reward}')
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
