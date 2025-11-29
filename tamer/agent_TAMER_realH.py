import numpy as np
import os
import pickle
import time
import random
import uuid
import cv2
from datetime import datetime
from itertools import count
from pathlib import Path
from csv import DictWriter
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

# Chemins
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')
MODELS_DIR.mkdir(exist_ok=True, parents=True)
LOGS_DIR.mkdir(exist_ok=True, parents=True)

# --- 1. Action Discretization ---
ACTIONS = [
    np.array([0.0, 0.0]),   # 0: Stop
    np.array([0.0, -0.5]),  # 1: Avant
    np.array([0.0, 0.5]),   # 2: Arrière
    np.array([-0.5, 0.0]),  # 3: Gauche
    np.array([0.5, 0.0])    # 4: Droite
]
ACTION_MAP = {0: 'Stop', 1: 'Avant', 2: 'Arriere', 3: 'Gauche', 4: 'Droite'}

class SGDFunctionApproximator:
    """Approximation de fonction avec RBF et SGD."""
    def __init__(self, env):
        observation_examples = []
        print("Initialisation du scaler (Sampling state space)...")
        obs, _ = env.reset()
        observation_examples.append(obs)
        
        for _ in range(200):
            action = env.action_space.sample() 
            
            if np.random.rand() < 0.05:
                 obs, _ = env.reset()
            else:

                 rand_action = np.random.uniform(-1, 1, size=(2,))
                 obs, _, done, _, _ = env.step(rand_action)
                 if done:
                     obs, _ = env.reset()
            
            observation_examples.append(obs)
        
        observation_examples = np.array(observation_examples, dtype='float64')
        
        self.scaler = StandardScaler()
        self.scaler.fit(observation_examples)

        self.featurizer = FeatureUnion([
            ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
            ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
            ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
            ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(len(ACTIONS)):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(observation_examples[0])], [0])
            self.models.append(model)
            
        print("Scaler initialisé.")

    def featurize_state(self, state):
        scaled = self.scaler.transform([state])
        return self.featurizer.transform(scaled)[0]

    def predict(self, state, action_index=None):
        features = self.featurize_state(state)
        if action_index is None:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[action_index].predict([features])[0]

    def update(self, state, action_index, td_target):
        features = self.featurize_state(state)
        self.models[action_index].partial_fit([features], [td_target])

class Tamer:
    """ Agent Hybride TAMER + RL pour FauteuilEnv """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=0.99, # Pour le Q-Learning
        epsilon=0.1,          # Exploration pour le Q-Learning
        min_eps=0.01,
        tame=True,
        ts_len=0.3,
        output_dir=LOGS_DIR,
        model_file_to_load=None
    ):
        self.env = env
        self.tame = tame
        self.ts_len = ts_len
        self.output_dir = output_dir
        self.num_episodes = num_episodes
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.min_eps = min_eps
        self.epsilon_step = (epsilon - min_eps) / num_episodes
        self.uuid = uuid.uuid4()

        if model_file_to_load is not None:
            self.load_model(model_file_to_load)
        else:
            self.H = SGDFunctionApproximator(env) # Modèle Humain (TAMER)
            self.Q = SGDFunctionApproximator(env) # Modèle Environnement (Q-Learning)

        # Logging
        self.reward_log_columns = ['Episode', 'Timestep', 'Feedback ts', 'Human Reward', 'Env Reward']
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

        self.metrics_log_columns = [
            'Episode', 'Total Reward', 'Total Feedback', 'Loss',
            'Collisions', 'Goal Reached', 'Intimate Zone Violations',
            'Personal Zone Violations', 'Social Zone Violations'
        ]
        self.metrics_log_path = os.path.join(self.output_dir, 'metrics_log_realH.csv')

        self.collision_count = 0
        self.goal_reached = 0
        self.intimate_violations = 0
        self.personal_violations = 0
        self.social_violations = 0
        self.public_violations = 0

    def act(self, state):
        """ Politique Hybride: argmax(H + Q) """
        if np.random.random() < 1 - self.epsilon:
            preds_H = self.H.predict(state) # Prédiction TAMER
            preds_Q = self.Q.predict(state) # Prédiction Q-Learning
            weight_H = 50.0 
            combined_preds = (preds_H * weight_H) + preds_Q

            return np.argmax(combined_preds)
        else:
            return np.random.randint(0, len(ACTIONS))

    # --- [NEW] Ajout de la méthode de pré-entraînement ---
    def pretrain(self, demonstration_data):
        """ Apprentissage supervisé à partir des démonstrations (Behavioral Cloning) """
        print(f"Start Pre-training on {len(demonstration_data)} samples...")
        data = list(demonstration_data)
        
        for epoch in range(20): 
            random.shuffle(data) 
            for state, action_idx in data:

                self.H.update(state, action_idx, 1.0)
                
                for other_action in range(len(ACTIONS)):
                    if other_action != action_idx:
                        self.H.update(state, other_action, -0.5)
                        
        print("Pre-training Done (Contrastive Mode).")
    # -----------------------------------------------------

    def _show_env(self, window_name='Simulation Fauteuil'):
        frame_rgb = self.env.render()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_bgr = cv2.resize(frame_bgr, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow(window_name, frame_bgr)
        return cv2.waitKey(1) & 0xFF

    def _update_metrics(self, state):
        robot_pos = state[:2]
        goal_pos = state[2:4]
        
        if np.linalg.norm(robot_pos - goal_pos) < 0.5:
            self.goal_reached = 1

        for human in self.env.humans:
            dist = np.linalg.norm(robot_pos - human["pos"])
            if self.env._is_in_field_of_view(robot_pos, goal_pos, human["pos"]):
                if dist < 0.4: self.intimate_violations += 1
                elif dist < 1.2: self.personal_violations += 1
                elif dist < 2.0: self.social_violations += 1
                else: self.public_violations += 1

    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}')
        state, _ = self.env.reset()
        
        self.collision_count = 0
        self.goal_reached = 0
        self.intimate_violations = 0
        self.personal_violations = 0
        self.social_violations = 0
        total_reward = 0
        total_feedback = 0
        
        cv2.namedWindow('Simulation Fauteuil', cv2.WINDOW_AUTOSIZE)

        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            if episode_index == 0: dict_writer.writeheader()

            for ts in count():
                action_idx = self.act(state)
                continuous_action = ACTIONS[action_idx]
                
                if self.tame:
                    disp.show_action(action_idx)

                next_state, env_reward, done, _, _ = self.env.step(continuous_action)
                
                # Mise à jour Q-Learning (Toujours)
                if done:
                    q_target = env_reward
                else:
                    q_next = np.max(self.Q.predict(next_state))
                    q_target = env_reward + self.discount_factor * q_next
                
                self.Q.update(state, action_idx, q_target)

                total_reward += env_reward
                if env_reward <= -10.0:
                    self.collision_count += 1
                self._update_metrics(next_state)

                key = self._show_env()
                if key == ord('q'):
                    print("Arrêt manuel (Q).")
                    done = True

                human_reward = 0
                if self.tame and not done:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        if self._show_env() == ord('q'):
                            done = True
                            break
                            
                        human_reward = disp.get_scalar_feedback()
                        if human_reward != 0:
                            self.H.update(state, action_idx, human_reward)
                            total_feedback += human_reward
                            dict_writer.writerow({
                                'Episode': episode_index + 1,
                                'Timestep': ts,
                                'Feedback ts': datetime.now(),
                                'Human Reward': human_reward,
                                'Env Reward': env_reward
                            })
                            break 

                if done:
                    print(f"  Episode terminé (Reward: {total_reward}, Collisions: {self.collision_count})")
                    cv2.destroyAllWindows()
                    break
                
                state = next_state

        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
            
        return total_reward, total_feedback

    def log_metrics(self, episode_index, total_reward, total_feedback):
        with open(self.metrics_log_path, 'a+', newline='') as metrics_obj:
            metrics_writer = DictWriter(metrics_obj, fieldnames=self.metrics_log_columns)
            if episode_index == 0:
                metrics_writer.writeheader()
            metrics_writer.writerow({
                'Episode': episode_index + 1,
                'Total Reward': total_reward,
                'Total Feedback': total_feedback,
                'Loss': 0,
                'Collisions': self.collision_count,
                'Goal Reached': self.goal_reached,
                'Intimate Zone Violations': self.intimate_violations,
                'Personal Zone Violations': self.personal_violations,
                'Social Zone Violations': self.social_violations
            })

    def train(self, model_filename='autosave_tamer'):
        from interface import Interface
        disp = Interface(action_map=ACTION_MAP)

        for i in range(self.num_episodes):
            total_reward, total_feedback = self._train_episode(i, disp)
            self.log_metrics(i, total_reward, total_feedback)

        self.save_model(model_filename)

    def play(self, n_episodes=1):
        """ Joue avec le modèle entraîné (Corrigé) """
        self.epsilon = 0 # Force l'exploitation pure
        cv2.namedWindow('Simulation Fauteuil', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Simulation Fauteuil', 800, 800)

        print(f"Début de la démonstration ({n_episodes} épisodes)...")
        all_episodes_data = []

        # Petit délai pour vider le buffer clavier et laisser l'utilisateur se préparer
        cv2.waitKey(1000) 

        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            print(f"Playing Episode {i+1}...")
            
            # [FIX 1] Initialiser la liste pour stocker la trajectoire
            episode_traj = []
            episode_traj.append(state[:2].copy())

            while not done:
                action_idx = self.act(state)
                continuous_action = ACTIONS[action_idx]
                state, _, done, _, _ = self.env.step(continuous_action)
                
                # [FIX 2] Enregistrer la position à chaque pas
                episode_traj.append(state[:2].copy())

                # [FIX 3] Augmenter le délai (30ms) pour voir le mouvement (sinon c'est instantané)
                key = self._show_env() 
                if key == ord('q'):
                    print("Arrêt manuel de la démonstration.")
                    self.env.close()
                    cv2.destroyAllWindows()
                    # Retourner ce qu'on a collecté jusqu'ici au lieu de rien
                    return all_episodes_data 
                
                # Petit sleep pour ralentir l'affichage visuel si waitKey ne suffit pas
                time.sleep(0.03)

            # [FIX 4] Sauvegarder les données de l'épisode complet
            all_episodes_data.append({
                'trajectory': np.array(episode_traj),
                'goal_pos': self.env.goal_pos,
                'objects': self.env.objects, # Pour l'affichage
                'humans': self.env.humans    # Pour l'affichage
            })

        print("Démonstration terminée.")
        self.env.close()
        cv2.destroyAllWindows()
        return all_episodes_data

    def save_model(self, filename):
        path = MODELS_DIR.joinpath(filename + '.p')
        models_to_save = {'H': self.H, 'Q': self.Q}
        with open(path, 'wb') as f:
            pickle.dump(models_to_save, f)
        print(f"Modèles H et Q sauvegardés : {path}")

    def load_model(self, filename):
        path = MODELS_DIR.joinpath(filename + '.p')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            self.H = data['H']
            self.Q = data['Q']
            print("Modèles H et Q chargés.")
        else:
            self.H = data
            self.Q = SGDFunctionApproximator(self.env)
            print("Ancien modèle H chargé, nouveau Q initialisé.")