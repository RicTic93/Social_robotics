import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class FauteuilEnv(gym.Env):
    def __init__(self, config):
        super(FauteuilEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Import config parameters
        self.config = config
        num_objects = self.config.get('num_objects', 3)
        self.social_distance = self.config.get('social_distance', 1.5)

        self.humans = []

        for group in self.config.get('static_groups', []):
            center_point = np.array(group.get('center_pos', [5.0, 5.0]))
            radius = group.get('radius', 2.0)
            count = group.get('count', 2)
            formation = group.get('formation', 'random')

            for i in range(count):
                angle = (2 * math.pi * i) / count
                human_pos = np.array([
                    center_point[0] + radius * math.cos(angle),
                    center_point[1] + radius * math.sin(angle)
                ])

                # Calculate orientation (converging/diverging)
                if formation == 'converging':
                    direction_vec = center_point - human_pos
                elif formation == 'diverging':
                    direction_vec = human_pos - center_point
                else:
                    direction_vec = np.random.uniform(-1, 1, size=2)
                
                # Add the static human to the list
                self.humans.append({
                    "type": "static", # <-- Important marker
                    "pos": human_pos,
                    "direction": self.normalize(direction_vec),
                })

        # B. Generate DYNAMIC HUMANS (from config)
        dynamic_count = self.config.get('dynamic_humans_count', 0)
        min_human_dist = 1.0

        for _ in range(dynamic_count):
            valid = False
            attempts = 0
            pos = np.array([0.0, 0.0]) # Position par défaut

            while not valid and attempts < 100:
                pos = np.random.uniform(0, 10, size=2)
                valid = True

                # Verify the distance with ALL already created humans
                # (includes static groups and previous dynamics)
                for human in self.humans:
                    if np.linalg.norm(pos - human["pos"]) < min_human_dist:
                        valid = False
                        break
                attempts += 1

            self.humans.append({
                "type": "dynamic", # <-- Important marker
                "pos": pos,
                "direction": self.normalize(np.random.uniform(-1, 1, size=2)),
                "duration": np.random.randint(30, 60)
            })


        total_num_humans = len(self.humans)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 3*num_objects + 2*total_num_humans,), dtype=np.float32)

        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_pos = np.array([8.0, 8.0], dtype=np.float32)
        self.max_speed = 0.5

        # Génère les obstacles en évitant les chevauchements
        self.objects = []
        min_distance_param = self.social_distance  # Distance entre les obstacles est editable
        human_radius = 0.5
        for _ in range(num_objects):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                pos = np.random.uniform(0, 10, size=2)
                radius = np.random.uniform(0.3, 0.8)
                valid = True

                for obj in self.objects:
                    # La distance doit être > rayons + distance sociale
                    dist_secu_obj = obj["radius"] + radius + min_distance_param
                    if np.linalg.norm(pos - obj["pos"]) < dist_secu_obj:
                        valid = False
                        break
                
                if not valid: 
                    attempts += 1
                    continue

                # 2. Vérifie les humains
                for human in self.humans:
                    # La distance doit être > rayons + distance sociale
                    dist_secu_human = human_radius + radius + min_distance_param
                    if np.linalg.norm(pos - human["pos"]) < dist_secu_human:
                        valid = False
                        break

                if valid:
                    self.objects.append({"pos": pos, "radius": radius})
                attempts += 1

        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Fauteuil Roulant Intelligent")
        self.clock = pygame.time.Clock()

    def normalize(self, v):
        """ Fonction utilitaire pour normaliser un vecteur (nouvelle fonction) """
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v
    
    def reset(self, seed=None, options=None):
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)

        human_radius = 0.5 # Doit correspondre à la collision dans step()
        min_social_dist = self.social_distance # Doit correspondre à move_humans()
        
        # Réinitialise les positions des humains
        for human in self.humans:
            if human["type"] == "dynamic":
                valid = False
                attempts = 0
                pos = np.array([0.0, 0.0]) # Position par défaut

                while not valid and attempts < 100:
                    pos = np.random.uniform(0, 10, size=2)
                    valid = True

                    # 1. Vérifie les obstacles
                    for obj in self.objects:
                        # Calcule la distance de sécurité (rayons + distance sociale)
                        dist_secu_obj = obj["radius"] + human_radius + min_social_dist
                        if np.linalg.norm(pos - obj["pos"]) < dist_secu_obj:
                            valid = False
                            break
                    
                    if not valid: 
                        attempts += 1
                        continue # Inutile de vérifier les humains

                    # 2. Vérifie les autres humains (statiques ET dynamiques)
                    for other in self.humans:
                        if other is human: # Ne pas se vérifier soi-même
                            continue
                        # Calcule la distance de sécurité (rayons + distance sociale)
                        dist_secu_human = human_radius + human_radius + min_social_dist
                        if np.linalg.norm(pos - other["pos"]) < dist_secu_human:
                            valid = False
                            break
                    attempts += 1

                human["pos"] = pos
                human["direction"] = self.normalize(np.random.uniform(-1, 1, size=2))
                human["duration"] = np.random.randint(30, 60)

        return self._get_obs(), {}

    def move_humans(self):
        for i, human in enumerate(self.humans):
            # Cette condition fait tout le travail :
            if human["type"] == "dynamic":
                # (Le reste de la logique est CONSERVÉ, mais indenté)
                if human["duration"] <= 0:
                    human["direction"] = self.normalize(np.random.uniform(-1, 1, size=2))
                    human["duration"] = np.random.randint(30, 60)

                new_pos = human["pos"] + human["direction"] * 0.05
                human["duration"] -= 1
                new_pos = np.clip(new_pos, 0, 10)

                valid = True
                # Vérifie la distance avec TOUS les autres humains (statiques ou dynamiques)
                for j, other_human in enumerate(self.humans):
                    if i != j and np.linalg.norm(new_pos - other_human["pos"]) < 0.5:
                        valid = False
                        # 1. Calcule le vecteur opposé (s'éloigner de l'autre humain)
                        away_vector = human["pos"] - other_human["pos"]
                        
                        # 2. Définit cette nouvelle direction
                        human["direction"] = self.normalize(away_vector)
                        
                        # 3. Donne-lui un peu de temps pour s'éloigner avant de re-changer
                        human["duration"] = 20 # (par exemple, 20 frames)
                        
                        break

                if valid:
                    for obj in self.objects:
                        if np.linalg.norm(new_pos - obj["pos"]) < 0.5 + obj["radius"]:
                            valid = False
                            break
                if valid:
                    self.humans[i]["pos"] = new_pos

    def step(self, action):
        self.robot_pos += action * self.max_speed
        self.robot_pos = np.clip(self.robot_pos, 0, 10)

        # Déplace les personnes
        self.move_humans()

        reward = -0.1
        terminated = False

        # Collision avec les obstacles
        for obj in self.objects:
            if np.linalg.norm(self.robot_pos - obj["pos"]) < 0.5 + obj["radius"]:
                reward = -10.0
                terminated = True
                break

        # Collision avec les personnes
        for human in self.humans:
            if np.linalg.norm(self.robot_pos - human["pos"]) < 0.5:
                reward = -10.0
                terminated = True
                break

        # But atteint
        if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5:
            reward = 10.0
            terminated = True

        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        obs = np.concatenate([self.robot_pos, self.goal_pos])
        for obj in self.objects:
            obs = np.concatenate([obs, obj["pos"], [obj["radius"]]])
        for human in self.humans:
            obs = np.concatenate([obs, human["pos"]])
        return obs

    def render(self):
        self.screen.fill((255, 255, 255))

        # Dessine le but (vert)
        pygame.draw.circle(
            self.screen,
            (0, 255, 0),
            (int(self.goal_pos[0] * 50) + 50, int(self.goal_pos[1] * 50) + 50),
            20
        )

        # Dessine les obstacles (bleu)
        for obj in self.objects:
            pygame.draw.circle(
                self.screen,
                (0, 0, 255),
                (int(obj["pos"][0] * 50) + 50, int(obj["pos"][1] * 50) + 50),
                int(obj["radius"] * 50)
            )

        # Dessine les personnes (vert + rouge pointe)
        for human in self.humans:
            pos = human["pos"]
            direction = human["direction"]
            size = 25 
            base_width = 15

            
            p1 = np.array([size / 2.0, 0]); p2 = np.array([-size / 2.0, base_width]); p3 = np.array([-size / 2.0, -base_width])
            angle_rad = math.atan2(direction[1], direction[0])
            rotation_matrix = np.array([[math.cos(angle_rad), -math.sin(angle_rad)], [math.sin(angle_rad), math.cos(angle_rad)]])
            p1_rot, p2_rot, p3_rot = p1 @ rotation_matrix.T, p2 @ rotation_matrix.T, p3 @ rotation_matrix.T

            scale, offset = 50, 50
            center_px = (pos * scale) + offset
            
            v1, v2, v3 = center_px + p1_rot, center_px + p2_rot, center_px + p3_rot
            
            color = (0, 150, 0)

            if human["type"] == "dynamic":
                color = (150, 0, 150)
            pygame.draw.polygon(self.screen, color, [v1, v2, v3])
            pygame.draw.circle(self.screen, (255, 0, 0), (int(v1[0]), int(v1[1])), 5)
            # Dessine le fauteuil (noir)
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (int(self.robot_pos[0] * 50) + 25, int(self.robot_pos[1] * 50) + 25, 50, 50)
            )

            pygame.draw.circle(
                self.screen,
                (255, 0, 0), 
                (int(v1[0]), int(v1[1])),5
            )
            

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
