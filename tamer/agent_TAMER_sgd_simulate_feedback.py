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

        # Initialisation des compteurs
        self.collision_count = 0
        self.goal_reached = 0
        self.intimate_violations = 0
        self.personal_violations = 0
        self.social_violations = 0
        self.public_violations = 0

        self.total_rewards = []
        self.total_feedbacks = []
        self.losses = []
        self.predictabilities = []
        self.politenesses = []

        self.intimate_violations_per_episode = []
        self.personal_violations_per_episode = []
        self.social_violations_per_episode = []
        self.public_violations_per_episode = []
        self.collisions_per_episode = []
        self.goals_per_episode = []
        

        #self.H = SGDFunctionApproximator(env, max_objects_in_view, max_humans_in_view)
        #assert hasattr(self.H, 'models'), "self.H.models n'est pas initialisé correctement"


        # Initialise le modèle
        if model_file_to_load is not None:
            self.load_model(model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env, max_objects_in_view, max_humans_in_view)

        # Configuration du logging
        self.reward_log_columns = [
            'Episode', 'Timestep', 'Human Feedback', 'Environment Reward',
            'Total Reward', 'Action', 'Loss', 'Epsilon'
        ]

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.reward_log_path = os.path.join(self.output_dir, f'tamer_rewards_log_{current_time}.csv')

        self.metrics_log_columns = [
            'Episode', 'Total Reward', 'Total Feedback', 'Loss',
            'Collisions', 'Goal Reached', 'Intimate Zone Violations',
            'Personal Zone Violations', 'Social Zone Violations',
            'Predictability', 'Politeness'
        ]
        self.metrics_log_path = os.path.join(self.output_dir, 'metrics_log.csv')

    def log_metrics(self, episode_index, total_reward, total_feedback, episode_loss):
        """Enregistre les métriques dans un fichier CSV."""
        with open(self.metrics_log_path, 'a+', newline='') as metrics_obj:
            metrics_writer = DictWriter(metrics_obj, fieldnames=self.metrics_log_columns)
            if episode_index == 0:
                metrics_writer.writeheader()
            metrics_writer.writerow({
                'Episode': episode_index + 1,
                'Total Reward': total_reward,
                'Total Feedback': total_feedback,
                'Loss': episode_loss,
                'Collisions': self.collision_count,
                'Goal Reached': self.goal_reached,
                'Intimate Zone Violations': self.intimate_violations,
                'Personal Zone Violations': self.personal_violations,
                'Social Zone Violations': self.social_violations
            })

    def act(self, state):
        """Politique epsilon-greedy pour un espace d'action continu."""
        assert len(state) == 54, f"State size is {len(state)}, expected 54"
        if np.random.random() < 1 - self.epsilon:
            action = np.random.uniform(-1, 1, size=self.env.action_space.shape)
            #print(f"Random Action: {action}")
        else:
            action = self.H.predict(state)
            #print(f"Predicted Action: {action}")
            
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

        # Compteurs pour les violations des zones proxémiques
        self.intimate_violations = 0
        self.personal_violations = 0
        self.social_violations = 0
        self.public_violations = 0
        
        # --- Feedback humain visibles ---
        for human in self.env.humans:
            human_pos = np.array(human["pos"])
            dist = np.linalg.norm(robot_pos - human["pos"])

            # Vérifie si l'humain est dans le champ de vision du robot
            if self.env._is_in_field_of_view(robot_pos, goal_pos, human_pos):
            # Vérifie si le robot est dans le champ de vision de l'humain (180°)
                if self.env._is_robot_in_human_fov(human["pos"], human["direction"], robot_pos):
                    if dist < 0.4:  # Zone intime
                        feedback -= 5.0
                        self.intimate_violations += 1
                    elif dist < 1.2:  # Zone personnelle
                        feedback -= 2.0
                        self.personal_violations += 1
                    elif dist < 2.0:  # Zone sociale
                        feedback += 1.0
                        self.social_violations += 1
                    else:
                        feedback += 2.0  # Zone publique
                        self.public_violations += 1
      
        return feedback

    def _train_episode(self, episode_index):
        """Entraîne l'agent pour un épisode."""
        state, _ = self.env.reset()
        total_reward = 0
        total_feedback = 0
        episode_loss = 0
        self.collision_count = 0
        self.intimate_violations = 0
        self.personal_violations = 0
        self.social_violations = 0
        self.goal_reached = 0
        
        with open(self.reward_log_path, 'a+', newline='') as reward_obj:
            reward_writer = DictWriter(reward_obj, fieldnames=self.reward_log_columns)
            if episode_index == 0:
                reward_writer.writeheader()

            trajectory = [state[:2].copy()]

            for ts in count():
                action = self.act(state)
                next_state, env_reward, done, _, info = self.env.step(action)
                trajectory.append(next_state[:2].copy())
                
                # Feedback automatique combiné
                human_feedback = self._get_human_feedback(next_state)
                total_reward += env_reward
                total_feedback += human_feedback

                # Calcul de la perte (loss)
                if self.tame:
                    for i in range(len(action)):
                        features = self.H.featurize_state(state)
                        predicted = self.H.models[i].predict([features])[0]
                        loss = (human_feedback - predicted) ** 2
                        episode_loss += loss
                        self.H.update(state, i, human_feedback)

                # Log des récompenses et métriques
                reward_writer.writerow({
                    'Episode': episode_index + 1,
                    'Timestep': ts,
                    'Human Feedback': human_feedback,
                    'Environment Reward': env_reward,
                    'Total Reward': total_reward,
                    'Action': action,
                    'Loss': loss if self.tame else 0,
                    'Epsilon': self.epsilon
                })

                if done:
                    break

                state = next_state
        
        trajectory = np.array(trajectory)
        predictability = self.compute_predictability(trajectory) if len(trajectory) > 1 else np.nan
        politeness = self.compute_politeness(
            self.intimate_violations,
            self.personal_violations,
            self.social_violations
        )

        with open(self.metrics_log_path, 'a+', newline='') as metrics_obj:
            metrics_writer = DictWriter(metrics_obj, fieldnames=self.metrics_log_columns)
            if episode_index == 0:
                metrics_writer.writeheader()
            metrics_writer.writerow({
                'Episode': episode_index + 1,
                'Total Reward': total_reward,
                'Total Feedback': total_feedback,
                'Loss': episode_loss,
                'Collisions': self.collision_count,
                'Goal Reached': self.goal_reached,
                'Intimate Zone Violations': self.intimate_violations,
                'Personal Zone Violations': self.personal_violations,
                'Social Zone Violations': self.social_violations,
                'Predictability': predictability,
                'Politeness': politeness
            })
        self.total_rewards.append(total_reward)
        self.total_feedbacks.append(total_feedback)
        self.losses.append(episode_loss)
        self.predictabilities.append(predictability)
        self.politenesses.append(politeness)
        self.intimate_violations_per_episode.append(self.intimate_violations)
        self.personal_violations_per_episode.append(self.personal_violations)
        self.social_violations_per_episode.append(self.social_violations)
        self.public_violations_per_episode.append(self.public_violations)
        self.collisions_per_episode.append(self.collision_count)
        self.goals_per_episode.append(self.goal_reached)


        # Décroissance d'epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

        return total_reward, total_feedback, episode_loss,predictability, politeness
    
    def compute_predictability(self,trajectory):
        """
        trajectory: np.array de shape (T,2)
        Retourne un score de lisibilité : plus le score est élevé, plus la trajectoire est prévisible
        """
        deltas = np.diff(trajectory, axis=0)
        angles = np.arctan2(deltas[:,1], deltas[:,0])
        heading_change = np.diff(angles)
        # Score = inverse de la variation moyenne des angles
        predictability_score = 1 / (1 + np.mean(np.abs(heading_change)))
        return predictability_score
    
    def compute_politeness(self,intimate, personal, social):
        """
        Score de politesse basé sur les violations des zones proxémiques.
        Plus le score est élevé, plus le robot est poli.
        """
        score = 1 / (1 + intimate + 0.5*personal + 0.1*social)
        return score
    
    def train(self, model_file_to_save=None):
        for i in range(self.num_episodes):
            total_reward, total_feedback, episode_loss, predictability, politeness = self._train_episode(i)
            print(f"Episode {i+1}: Total Reward = {total_reward}, Total Feedback = {total_feedback}, Loss = {episode_loss}")
            self.log_metrics(i, total_reward,total_feedback, episode_loss)
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
