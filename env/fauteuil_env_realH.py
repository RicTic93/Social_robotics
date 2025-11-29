import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math

class FauteuilEnv(gym.Env):
    def __init__(self, config):
        super(FauteuilEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # --- Configuration ---
        self.config = config
        self.num_objects = self.config.get('num_objects', 3)
        self.social_distance = self.config.get('social_distance', 1.5)
        self.static_groups_conf = self.config.get('static_groups', [])
        self.dynamic_humans_count = self.config.get('dynamic_humans_count', 0)
        
        # Paramètres FOV
        self.fov_distance = config.get('fov_distance', 5.0)
        self.human_fov_distance = config.get('human_fov_distance', 3.0)
        self.human_fov_angle = config.get('human_fov_angle', 180)

        # Calcul pour l'espace d'observation
        total_static = sum([g.get('count', 0) for g in self.static_groups_conf])
        total_humans = total_static + self.dynamic_humans_count
        self.total_num_humans = total_humans

        # Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(72,), dtype=np.float32)

        # Variables d'état
        self.humans = []
        self.objects = []
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_pos = np.array([8.0, 8.0], dtype=np.float32)
        self.max_speed = 0.5
        self.max_group_distance = 2.0
        
        # Flag pour régénérer la map
        self.regenerate_layout = True 
        self.training_mode = True
        # --- Pygame Init ---
        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        self.canvas = pygame.Surface((600, 600))
        pygame.display.set_caption("Fauteuil Roulant Intelligent")
        self.clock = pygame.time.Clock()

    def normalize(self, v):
        norm = np.linalg.norm(v)
        if norm > 0: return v / norm
        return v

    def _check_collision_free(self, pos, radius, exclude_list=[]):
        """ Vérification stricte de collision pour le spawn. """
        margin = 0.5
        if pos[0] - radius < margin or pos[0] + radius > 10 - margin or \
           pos[1] - radius < margin or pos[1] + radius > 10 - margin:
            return False

        if np.linalg.norm(pos - self.robot_pos) < 2.0: return False
        if np.linalg.norm(pos - self.goal_pos) < 2.0: return False

        for obj in self.objects:
            if obj in exclude_list: continue
            if np.linalg.norm(pos - obj["pos"]) < (obj["radius"] + radius + self.social_distance):
                return False

        for human in self.humans:
            if human in exclude_list: continue
            human_radius = 0.5 
            if np.linalg.norm(pos - human["pos"]) < (human_radius + radius + self.social_distance):
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
            
            valid_center = False
            center_pos = np.array([5.0, 5.0])
            attempts = 0
            while not valid_center and attempts < 100:
                center_pos = np.random.uniform(1, 9, size=2)
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
                if self._check_collision_free(pos, radius):
                    self.objects.append({"pos": pos, "radius": radius})
                    valid = True
                attempts += 1

    def _respawn_dynamic_humans(self):
        """ Régénère uniquement les humains dynamiques """
        self.humans = [h for h in self.humans if h["type"] == "static"]

        for _ in range(self.dynamic_humans_count):
            pos = np.array([0.0, 0.0])
            found = False
            search_stages = [
                {"margin": 1.5, "attempts": 200},
                {"margin": 0.5, "attempts": 500},
                {"margin": 0.01, "attempts": 1000}
            ]
            for stage in search_stages:
                if found: break
                margin = stage["margin"]
                max_attempts = stage["attempts"]
                attempts = 0
                while not found and attempts < max_attempts:
                    candidate_pos = np.random.uniform(0, 10, size=2)
                    if self._check_collision_free(candidate_pos, 0.5 + margin):
                        pos = candidate_pos
                        found = True
                    attempts += 1
            
            if found:
                self.humans.append({
                    "type": "dynamic",
                    "pos": pos,
                    "direction": self.normalize(np.random.uniform(-1, 1, size=2)),
                    "duration": np.random.randint(30, 60)
                })

    def set_training_mode(self, mode):
        self.training_mode = mode

    def reset(self, seed=None, options=None):
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        
        if self.regenerate_layout:
            self._generate_layout()
        
        self._respawn_dynamic_humans()
        self.regenerate_layout = False 
        
        full_obs = self._get_obs()
        partial_obs = self._get_partial_obs(full_obs)
        return partial_obs, {}

    # --- LOGIQUE FOV & SOCIALE ---
    def is_between_converging_humans(self):
        for i, h1 in enumerate(self.humans):
            if h1["type"] != "static": continue
            for j, h2 in enumerate(self.humans):
                if i >= j or h2["type"] != "static": continue
                
                if np.linalg.norm(h1["pos"] - h2["pos"]) < 3.0:
                    d1 = np.linalg.norm(self.robot_pos - h1["pos"])
                    d2 = np.linalg.norm(self.robot_pos - h2["pos"])
                    d_humans = np.linalg.norm(h1["pos"] - h2["pos"])
                    if abs((d1 + d2) - d_humans) < 0.2:
                        return True
        return False

    def _is_in_field_of_view(self, robot_pos, goal_pos, target_pos, angle_deg=140):
        distance = np.linalg.norm(target_pos - robot_pos)
        if distance > self.fov_distance: return False
        
        direction = goal_pos - robot_pos
        if np.linalg.norm(direction) == 0: return False
        
        target_dir = target_pos - robot_pos
        
        angle_robot = math.atan2(direction[1], direction[0])
        angle_target = math.atan2(target_dir[1], target_dir[0])
        
        angle_diff = (angle_target - angle_robot + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(angle_diff)) <= angle_deg / 2

    def _is_robot_in_human_fov(self, human_pos, human_direction, robot_pos, angle_deg=160):
        distance = np.linalg.norm(robot_pos - human_pos)
        if distance > self.human_fov_distance: return False
        
        target_dir = robot_pos - human_pos
        human_dir_angle = math.atan2(human_direction[1], human_direction[0])
        target_angle = math.atan2(target_dir[1], target_dir[0])
        
        angle_diff = (target_angle - human_dir_angle + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(angle_diff)) <= angle_deg / 2

    def _get_partial_obs(self, full_obs):
        # 74 features fixes
        # [MODIFICATION] Nous gardons les index 0-3 en absolu pour les calculs internes de l'agent (metrics)
        # Mais les objets détectés seront relatifs.
        partial_obs = np.zeros(74, dtype='float64')
        partial_obs[:2] = self.robot_pos # Position absolue (nécessaire pour le calcul de distance au but dans l'agent)
        partial_obs[2:4] = self.goal_pos # Position absolue
        
        # Objets visibles
        obj_idx = 0
        for obj in self.objects:
            if self._is_in_field_of_view(self.robot_pos, self.goal_pos, obj["pos"]):
                if obj_idx < 10:
                    base = 4 + 3*obj_idx
                    # [MODIFICATION IMPORTANTE 1] Coordonnées relatives pour une meilleure généralisation
                    # Au lieu d'apprendre "l'obstacle est en (8,8)", on apprend "l'obstacle est à (0.5, 0) de moi"
                    relative_pos = obj["pos"] - self.robot_pos
                    partial_obs[base:base+2] = relative_pos
                    partial_obs[base+2] = obj["radius"]
                    obj_idx += 1
        
        # Humains visibles
        human_idx = 0
        for human in self.humans:
            if self._is_in_field_of_view(self.robot_pos, self.goal_pos, human["pos"]):
                if human_idx < 10:
                    base = 34 + 4*human_idx
                    # [MODIFICATION IMPORTANTE 1] Coordonnées relatives
                    relative_pos = human["pos"] - self.robot_pos
                    partial_obs[base:base+2] = relative_pos
                    partial_obs[base+2:base+4] = human["direction"]
                    human_idx += 1
        return partial_obs

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
                for obj in self.objects:
                    if np.linalg.norm(new_pos - obj["pos"]) < 0.5 + obj["radius"]:
                        valid_move = False
                        away = human["pos"] - obj["pos"]
                        human["direction"] = self.normalize(away)
                        human["duration"] = 20
                        break
                
                if valid_move:
                    for j, other in enumerate(self.humans):
                        if i == j: continue
                        if np.linalg.norm(new_pos - other["pos"]) < 1.0:
                            valid_move = False
                            away = human["pos"] - other["pos"]
                            human["direction"] = self.normalize(away)
                            human["duration"] = 20
                            break
                if valid_move:
                    human["pos"] = new_pos

    def step(self, action):
        dist_old = np.linalg.norm(self.robot_pos - self.goal_pos)

        self.robot_pos += action * self.max_speed
        self.robot_pos = np.clip(self.robot_pos, 0, 10)
        self.move_humans()

        dist_new = np.linalg.norm(self.robot_pos - self.goal_pos)

        reward = 0.0
        human_feedback = 0.0
        terminated = False

        # Reward Shaping
        progress = dist_old - dist_new
        reward += progress * 2.0 
        reward -= 0.05

        # [MODIFICATION IMPORTANTE 2] Safety Distance Penalty
        # Pénalité préventive si on s'approche trop d'un obstacle (Warning Zone)
        # Cela évite de devoir "taper" l'obstacle pour avoir un retour négatif
        safety_margin = 0.4 # 30cm de marge de sécurité autour du rayon physique
        
        for obj in self.objects:
            d_obj = np.linalg.norm(self.robot_pos - obj["pos"])
            physical_limit = 0.5 + obj["radius"]
            # Si on est dans la zone "orange" (trop près mais pas encore crash)
            if physical_limit < d_obj < physical_limit + safety_margin:
                reward -= 2.0 # Avertissement : "Tu es trop près !"

        # Feedback Social
        for human in self.humans:
            dist = np.linalg.norm(self.robot_pos - human["pos"])
            if self._is_in_field_of_view(self.robot_pos, self.goal_pos, human["pos"]):
                if self._is_robot_in_human_fov(human["pos"], human["direction"], self.robot_pos):
                    if dist < 0.4: human_feedback -= 5.0
                    elif dist < 1.2: human_feedback -= 2.0
                    elif dist < 2.0: human_feedback += 1.0
                    else: human_feedback += 2.0

        # Collision environnementale
        for human in self.humans:
            if np.linalg.norm(self.robot_pos - human["pos"]) < 0.5:
                if not self._is_robot_in_human_fov(human["pos"], human["direction"], self.robot_pos):
                    reward -= 100.0
                else:
                    reward -= 50.0
                
                if self.training_mode:
                    terminated = False
                else:
                    terminated = True
                    self.regenerate_layout = False
                break
        
        if not terminated:
            for obj in self.objects:
                if np.linalg.norm(self.robot_pos - obj["pos"]) < 0.5 + obj["radius"]:
                    reward -= 100.0
                    if self.training_mode:
                        terminated = False
                    else:
                        terminated = True
                        self.regenerate_layout = False
                    break
        
        margin = 0.2
        if self.robot_pos[0] < margin or self.robot_pos[0] > 10-margin or \
           self.robot_pos[1] < margin or self.robot_pos[1] > 10-margin:
            reward -= 3.0

        # Zone A: < 0.6m
        if self.robot_pos[0] < 0.6 or self.robot_pos[0] > 9.4 or \
           self.robot_pos[1] < 0.6 or self.robot_pos[1] > 9.4:
            reward -= 2.0

        # Zone B: < 0.2m
        if self.robot_pos[0] < 0.2 or self.robot_pos[0] > 9.8 or \
           self.robot_pos[1] < 0.2 or self.robot_pos[1] > 9.8:
            reward -= 5.0

        if self.is_between_converging_humans():
            reward -= 20.0

        # Succès
        if not terminated:
            if np.linalg.norm(self.robot_pos - self.goal_pos) < 0.8:
                reward += 20.0 
                terminated = True
                self.regenerate_layout = True

        full_obs = self._get_obs()
        info = {"human_feedback": human_feedback}
        return self._get_partial_obs(full_obs), reward, terminated, False, info

    def _get_obs(self):
        obs = np.concatenate([self.robot_pos, self.goal_pos])
        for obj in self.objects:
            obs = np.concatenate([obs, obj["pos"], [obj["radius"]]])
        for human in self.humans:
            obs = np.concatenate([obs, human["pos"]])
        return obs

    # --- RENDERING ---
    def _draw_human_fov(self, surface, human_pos, human_direction, color=(255, 0, 0, 100)):
        center_px = (int(human_pos[0] * 50) + 50, int(human_pos[1] * 50) + 50)
        direction_angle = math.atan2(human_direction[1], human_direction[0])
        fov_surface = pygame.Surface((600, 600), pygame.SRCALPHA)
        half_angle = math.radians(self.human_fov_angle / 2)
        start_angle = direction_angle - half_angle
        end_angle = direction_angle + half_angle
        
        points = [center_px]
        num_points = 10
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center_px[0] + self.human_fov_distance * 50 * math.cos(angle)
            y = center_px[1] + self.human_fov_distance * 50 * math.sin(angle)
            points.append((x, y))
        
        if len(points) > 2:
            pygame.draw.polygon(fov_surface, color, points)
        surface.blit(fov_surface, (0, 0))

    def _draw_fov(self, surface, center, direction, angle_deg=140, color=(0, 255, 255, 100)):
        center_px = (int(center[0] * 50) + 50, int(center[1] * 50) + 50)
        direction_angle = math.atan2(direction[1], direction[0])
        fov_surface = pygame.Surface((600, 600), pygame.SRCALPHA)
        half_angle = math.radians(angle_deg / 2)
        start_angle = direction_angle - half_angle
        end_angle = direction_angle + half_angle
        
        points = [center_px]
        num_points = 10
        for i in range(num_points + 1):
            angle = start_angle + (end_angle - start_angle) * i / num_points
            x = center_px[0] + self.fov_distance * 50 * math.cos(angle)
            y = center_px[1] + self.fov_distance * 50 * math.sin(angle)
            points.append((x, y))
            
        if len(points) > 2:
            pygame.draw.polygon(fov_surface, color, points)
        surface.blit(fov_surface, (0, 0))

    def render(self):
        self.canvas.fill((255, 255, 255))
        pygame.draw.circle(self.canvas, (0, 255, 0), (int(self.goal_pos[0]*50)+50, int(self.goal_pos[1]*50)+50), 20)
        for obj in self.objects:
            pygame.draw.circle(self.canvas, (0, 0, 255), (int(obj["pos"][0]*50)+50, int(obj["pos"][1]*50)+50), int(obj["radius"]*50))
        for human in self.humans:
            pos = human["pos"]; direction = human["direction"]
            self._draw_human_fov(self.canvas, pos, direction)
            size, base = 25, 15
            angle = math.atan2(direction[1], direction[0])
            rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            p1 = np.array([size/2, 0]) @ rot.T; p2 = np.array([-size/2, base]) @ rot.T; p3 = np.array([-size/2, -base]) @ rot.T
            center_px = (pos*50)+50
            v1,v2,v3 = center_px+p1, center_px+p2, center_px+p3
            color = (0, 150, 0) if human["type"] == "static" else (150, 0, 150)
            pygame.draw.polygon(self.canvas, color, [v1, v2, v3])
            pygame.draw.circle(self.canvas, (255, 0, 0), (int(v1[0]), int(v1[1])), 5)
        direction = self.goal_pos - self.robot_pos
        self._draw_fov(self.canvas, self.robot_pos, direction)
        pygame.draw.rect(self.canvas, (0, 0, 0), (int(self.robot_pos[0]*50)+25, int(self.robot_pos[1]*50)+25, 30, 30))
        if self.is_between_converging_humans():
            pygame.draw.circle(self.canvas, (255, 0, 0), (int(self.robot_pos[0]*50)+25, int(self.robot_pos[1]*50)+25), 20, 2)
        try:
            current_surface = pygame.display.get_surface()
            if current_surface is not None:
                w, h = current_surface.get_size()
                if w == 600 and h == 600:
                    self.screen.blit(self.canvas, (0, 0))
                    pygame.display.flip()
        except: pass
        return np.transpose(pygame.surfarray.array3d(self.canvas), (1, 0, 2))

    def close(self):
        if self.screen is not None: 
            pass