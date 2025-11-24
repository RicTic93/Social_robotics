import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class FauteuilEnv(gym.Env):
    def __init__(self, config):
        super(FauteuilEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Import parametre de config
        self.config = config
        self.num_objects = self.config.get('num_objects', 3)
        self.social_distance = self.config.get('social_distance', 1.5)
        # On stocke la config des groupes statiques pour pouvoir les régénérer plus tard
        self.static_groups_conf = self.config.get('static_groups', [])
        self.dynamic_humans_count = self.config.get('dynamic_humans_count', 0)

        # Calcul pour l'espace d'observation
        total_static = sum([g.get('count', 0) for g in self.static_groups_conf])
        total_humans = total_static + self.dynamic_humans_count
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 3*self.num_objects + 2*total_humans,), dtype=np.float32)

        # Variables d'état
        self.humans = []
        self.objects = []
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_pos = np.array([8.0, 8.0], dtype=np.float32)
        self.max_speed = 0.5
        
        # Flag pour contrôler la régénération de la map
        self.regenerate_layout = True # Au début, on génère tout

        # Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Fauteuil Roulant Intelligent")
        self.clock = pygame.time.Clock()

    def normalize(self, v):
        """ Fonction utilitaire pour normaliser un vecteur (nouvelle fonction) """
        norm = np.linalg.norm(v)
        if norm > 0: return v / norm
        return v

    def _check_collision_free(self, pos, radius, exclude_list=[]):
        """
        Vérifie si une position (cercle) est libre de toute collision.
        Utilise les rayons réels + distance sociale.
        """
        # 1. Limites du monde (Marge)
        margin = 0.5
        if pos[0] - radius < margin or pos[0] + radius > 10 - margin or \
           pos[1] - radius < margin or pos[1] + radius > 10 - margin:
            return False

        # 2. Robot et But (Zones interdites)
        if np.linalg.norm(pos - self.robot_pos) < 2.0: return False
        if np.linalg.norm(pos - self.goal_pos) < 2.0: return False

        # 3. Obstacles
        for obj in self.objects:
            if obj in exclude_list: continue
            dist = np.linalg.norm(pos - obj["pos"])
            # Distance critique = Rayon Obj + Mon Rayon + Distance Sociale
            if dist < (obj["radius"] + radius + self.social_distance):
                return False

        # 4. Humains
        for human in self.humans:
            if human in exclude_list: continue
            dist = np.linalg.norm(pos - human["pos"])
            human_radius = 0.5 
            if dist < (human_radius + radius + self.social_distance):
                return False
        
        return True

    def _generate_layout(self):
        """ Génère le décor (Obstacles + Groupes Statiques) """
        self.humans = []
        self.objects = []
        
        # A. Groupes Statiques
        for group in self.static_groups_conf:
            count = group.get('count', 2)
            formation = group.get('formation', 'random')
            g_radius = group.get('radius', 2.0)
            
            # Trouver un centre valide pour le groupe (aléatoire)
            valid_center = False
            center_pos = np.array([5.0, 5.0])
            attempts = 0
            while not valid_center and attempts < 100:
                center_pos = np.random.uniform(1, 9, size=2)
                # On vérifie une zone large pour le groupe (rayon + marge)
                if self._check_collision_free(center_pos, g_radius): 
                    valid_center = True
                attempts += 1

            for i in range(count):
                angle = (2 * math.pi * i) / count
                h_pos = np.array([
                    center_pos[0] + g_radius * math.cos(angle),
                    center_pos[1] + g_radius * math.sin(angle)
                ])
                
                if formation == 'converging': dir_vec = center_pos - h_pos
                elif formation == 'diverging': dir_vec = h_pos - center_pos
                else: dir_vec = np.random.uniform(-1, 1, size=2)

                self.humans.append({
                    "type": "static",
                    "pos": h_pos,
                    "direction": self.normalize(dir_vec),
                    "duration": 0
                })

        # B. Obstacles
        for _ in range(self.num_objects):
            valid = False
            attempts = 0
            while not valid and attempts < 200:
                pos = np.random.uniform(0, 10, size=2)
                radius = np.random.uniform(0.3, 0.8)
                # Vérification stricte via _check_collision_free
                if self._check_collision_free(pos, radius):
                    self.objects.append({"pos": pos, "radius": radius})
                    valid = True
                attempts += 1

    def _respawn_dynamic_humans(self):
        """ 
        Régénère uniquement les humains dynamiques avec une stratégie de placement robuste.
        Essaie d'abord de trouver une position très large, puis réduit les contraintes si nécessaire.
        GARANTIT qu'aucun chevauchement physique ne se produit.
        """
        # On garde seulement les statiques
        self.humans = [h for h in self.humans if h["type"] == "static"]

        for _ in range(self.dynamic_humans_count):
            pos = np.array([0.0, 0.0])
            found = False
            
            # Stratégie de "Repli" (Fallback Strategy)
            # Niveau 1 : Idéal (Très large espace, 1.5m de marge extra)
            # Niveau 2 : Acceptable (Marge standard, 0.5m extra)
            # Niveau 3 : Critique (Juste éviter la collision physique, 0.05m extra)
            search_stages = [
                {"margin": 0.1, "attempts": 200},
                {"margin": 0.05, "attempts": 500},
                {"margin": 0.01, "attempts": 1000} # Beaucoup d'essais pour le dernier recours
            ]

            for stage in search_stages:
                if found: break
                
                margin = stage["margin"]
                max_attempts = stage["attempts"]
                attempts = 0
                
                while not found and attempts < max_attempts:
                    candidate_pos = np.random.uniform(0, 10, size=2)
                    
                    # On vérifie si la position est libre avec le rayon de l'humain (0.5) + la marge
                    if self._check_collision_free(candidate_pos,  margin):
                        pos = candidate_pos
                        found = True
                    attempts += 1
            
            # Si vraiment impossible (map saturée), on ne crée PAS cet humain
            # plutôt que de le faire spawner sur quelqu'un d'autre.
            if found:
                self.humans.append({
                    "type": "dynamic",
                    "pos": pos,
                    "direction": self.normalize(np.random.uniform(-1, 1, size=2)),
                    "duration": np.random.randint(30, 60)
                })
            else:
                print("Warning: Map trop encombrée, un humain dynamique n'a pas pu spawner.")
    def is_between_converging_humans(self):  # !!!!!!!!
        #max_group_distance = 2.0  # Distance maximale entre le fauteuil et le centre du groupe pour activer la détection
        converging_humans = []

        # Identifie les humains des groupes convergents
        for group in self.config.get("static_groups", []):
            if group["formation"] == "converging":
                center = np.array(group["center_pos"])
                # Vérifie si le fauteuil est proche du groupe
                if np.linalg.norm(self.robot_pos - center) > self.max_group_distance:
                    continue  # Ignore les groupes trop éloignés
                # Ajoute les humains de ce groupe à la liste
                for i in range(group["count"]):
                    angle = (2 * math.pi * i) / group["count"]
                    human_pos = np.array([
                        center[0] + group["radius"] * math.cos(angle),
                        center[1] + group["radius"] * math.sin(angle)
                    ])
                    # Trouve l'humain correspondant dans self.humans (tolérance pour les floats)
                    for h in self.humans:
                        if h["type"] == "static" and np.allclose(h["pos"], human_pos, atol=0.1):
                            converging_humans.append(h)

        # Vérifie si le fauteuil est entre deux de ces humains
        for i, h1 in enumerate(converging_humans):
            for j, h2 in enumerate(converging_humans):
                if i >= j:
                    continue
                d1 = np.linalg.norm(self.robot_pos - h1["pos"])
                d2 = np.linalg.norm(self.robot_pos - h2["pos"])
                d_humans = np.linalg.norm(h1["pos"] - h2["pos"])
                # Tolérance réduite pour éviter les faux positifs
                if abs((d1 + d2) - d_humans) < 0.2:  # Tolérance plus stricte
                    return True
        return False
    
    # agent can see in 140° angle field of view
    def _get_partial_obs(self, full_obs):
        robot_pos = full_obs[:2]
        goal_pos = full_obs[2:4]
        partial_obs = np.concatenate([robot_pos, goal_pos])

        # Filtrez les objets et humains dans le champ de vision
        for obj in self.objects:
            if self._is_in_field_of_view(robot_pos, obj["pos"]):
                partial_obs = np.concatenate([partial_obs, obj["pos"], [obj["radius"]]])
        for human in self.humans:
            if self._is_in_field_of_view(robot_pos, human["pos"]):
                partial_obs = np.concatenate([partial_obs, human["pos"]])

        return partial_obs

    # Obstacle and human in teh field of view
    def _is_in_field_of_view(self, robot_pos, target_pos, angle_deg=140):
        # Calculez l'angle entre la direction du fauteuil et la cible
        direction = self.goal_pos - robot_pos
        target_dir = target_pos - robot_pos
        angle = math.degrees(math.atan2(target_dir[1], target_dir[0]) - math.atan2(direction[1], direction[0]))
        return abs(angle) <= angle_deg / 2
    
    # robot dans le champ de vision des humains
    def _is_robot_in_human_fov(self, human_pos, human_direction, robot_pos, angle_deg=180):
        """
        Vérifie si le fauteuil est dans le champ de vision (180°) d'un humain.
        :param human_pos: Position de l'humain.
        :param human_direction: Direction vers laquelle l'humain regarde.
        :param robot_pos: Position du fauteuil.
        :param angle_deg: Angle de vision de l'humain (180° par défaut).
        :return: True si le fauteuil est dans le champ de vision de l'humain.
        """
        target_dir = robot_pos - human_pos
        angle = math.degrees(math.atan2(target_dir[1], target_dir[0]) - math.atan2(human_direction[1], human_direction[0]))
        return abs(angle) <= angle_deg / 2

    def reset(self, seed=None, options=None):
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        
        # --- LOGIQUE DE RÉGÉNÉRATION ---
        # Si c'était un SUCCÈS (ou le tout début), on régénère le décor
        if self.regenerate_layout:
            self._generate_layout()
        
        # Toujours régénérer les dynamiques pour éviter le spawn kill
        self._respawn_dynamic_humans()

        # Reset du flag (sera mis à True seulement si on gagne)
        self.regenerate_layout = False 

        return self._get_obs(), {}

    def move_humans(self):
        for i, human in enumerate(self.humans):
            if human["type"] == "dynamic":
                if human["duration"] <= 0:
                    human["direction"] = self.normalize(np.random.uniform(-1, 1, size=2))
                    human["duration"] = np.random.randint(30, 60)

                new_pos = human["pos"] + human["direction"] * 0.05
                human["duration"] -= 1
                new_pos = np.clip(new_pos, 0, 10)

                valid_move = True
                
                # 1. Vérif Obstacles
                for obj in self.objects:
                    if np.linalg.norm(new_pos - obj["pos"]) < 0.5 + obj["radius"]:
                        valid_move = False
                        # (Optionnel : Rebond ici aussi)
                        break
                
                # 2. Vérif Autres Humains
                if valid_move:
                    for j, other in enumerate(self.humans):
                        if i == j: continue
                        if np.linalg.norm(new_pos - other["pos"]) < 1.0: # 0.5+0.5
                            valid_move = False
                            
                            # --- Ta logique de rebond (conservée) ---
                            away = human["pos"] - other["pos"]
                            human["direction"] = self.normalize(away)
                            human["duration"] = 20
                            break
                
                if valid_move:
                    human["pos"] = new_pos
                    

    def step(self, action):
        self.robot_pos += action * self.max_speed
        self.robot_pos = np.clip(self.robot_pos, 0, 10)

        self.move_humans()

        reward = -0.1
        terminated = False

        # Collision Obstacles
        for obj in self.objects:
            if np.linalg.norm(self.robot_pos - obj["pos"]) < 0.5 + obj["radius"]:
                reward = -10.0
                terminated = True
                self.regenerate_layout = False # ÉCHEC : On garde la map
                break
        
        # Collision Humains
        if not terminated:
            for human in self.humans:
                if np.linalg.norm(self.robot_pos - human["pos"]) < 0.5:
                    reward = -10.0
                    terminated = True
                    self.regenerate_layout = False # ÉCHEC : On garde la map
                    break

        # Succès
        if not terminated:
            if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.5:
                reward = 10.0
                terminated = True
                self.regenerate_layout = True # SUCCÈS : Nouvelle map !

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
        pygame.draw.circle(self.screen, (0, 255, 0), (int(self.goal_pos[0]*50)+50, int(self.goal_pos[1]*50)+50), 20)
        for obj in self.objects:
            pygame.draw.circle(self.screen, (0, 0, 255), (int(obj["pos"][0]*50)+50, int(obj["pos"][1]*50)+50), int(obj["radius"]*50))
        for human in self.humans:
            pos = human["pos"]; direction = human["direction"]
            size, base = 25, 15
            angle = math.atan2(direction[1], direction[0])
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            p1 = np.array([size/2, 0]) @ rot.T; p2 = np.array([-size/2, base]) @ rot.T; p3 = np.array([-size/2, -base]) @ rot.T
            center_px = (pos*50)+50
            v1,v2,v3 = center_px+p1, center_px+p2, center_px+p3
            color = (0, 150, 0) if human["type"] == "static" else (150, 0, 150)
            pygame.draw.polygon(self.screen, color, [v1, v2, v3])
            pygame.draw.circle(self.screen, (255, 0, 0), (int(v1[0]), int(v1[1])), 5)
        #visualition robot
        pygame.draw.rect(self.screen, (0, 0, 0), (int(self.robot_pos[0]*50)+25, int(self.robot_pos[1]*50)+25, 50, 50))
        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None: pygame.quit()