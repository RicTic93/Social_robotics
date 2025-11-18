import gymnasium as gym
from gymnasium import spaces
from config import config
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

        # Generate obstacles while avoiding overlaps
        self.objects = []
        min_distance_param = self.social_distance  # Distance between obstacles is editable
        human_radius = 0.5
        for _ in range(num_objects):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                pos = np.random.uniform(0, 10, size=2)
                radius = np.random.uniform(0.3, 0.8)
                valid = True
                
                 # Vérifie la distance avec le point de départ du fauteuil
                if np.linalg.norm(pos - self.robot_pos) < radius + 0.5 + self.social_distance:
                    valid = False

                # Vérifie la distance avec le goal
                if np.linalg.norm(pos - self.goal_pos) < radius + 0.5 + self.social_distance:
                    valid = False

                for obj in self.objects:
                    # The distance must be > radii + social distance
                    dist_secu_obj = obj["radius"] + radius + min_distance_param
                    if np.linalg.norm(pos - obj["pos"]) < dist_secu_obj:
                        valid = False
                        break
                
                if not valid: 
                    attempts += 1
                    continue

                # 2. Verify the humans
                for human in self.humans:
                    # The distance must be > radii + social distance
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

        human_radius = 0.5 # Must correspond to collision in step()
        min_social_dist = self.social_distance # Must correspond to move_humans()
        
        # Reset human positions
        for human in self.humans:
            if human["type"] == "dynamic":
                valid = False
                attempts = 0
                pos = np.array([0.0, 0.0]) # Default position

                while not valid and attempts < 100:
                    pos = np.random.uniform(0, 10, size=2)
                    valid = True

                    # 1. Verify the obstacles
                    for obj in self.objects:
                        # Calculate the safety distance (radii + social distance)
                        dist_secu_obj = obj["radius"] + human_radius + min_social_dist
                        if np.linalg.norm(pos - obj["pos"]) < dist_secu_obj:
                            valid = False
                            break
                    
                    if not valid: 
                        attempts += 1
                        continue # No need to check humans

                    # 2. Verify the other humans (static AND dynamic)
                    for other in self.humans:
                        if other is human: # Do not check oneself
                            continue
                        # Calculate the safety distance (radii + social distance)
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
            # This condition does all the work:
            if human["type"] == "dynamic":
                # (The rest of the logic is PRESERVED, but indented)
                if human["duration"] <= 0:
                    human["direction"] = self.normalize(np.random.uniform(-1, 1, size=2))
                    human["duration"] = np.random.randint(30, 60)

                new_pos = human["pos"] + human["direction"] * 0.05
                human["duration"] -= 1
                new_pos = np.clip(new_pos, 0, 10)

                valid = True
                # Verify the distance with ALL other humans (static or dynamic)
                for j, other_human in enumerate(self.humans):
                    if i != j and np.linalg.norm(new_pos - other_human["pos"]) < 0.5:
                        valid = False
                        # 1. Calculate the opposite vector (move away from the other human)
                        away_vector = human["pos"] - other_human["pos"]
                        
                        # 2. Set this new direction
                        human["direction"] = self.normalize(away_vector)
                        
                        # 3. Give it some time to move away before changing again
                        human["duration"] = 20 # (for example, 20 frames)
                        
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

        # Move the humans
        self.move_humans()

        reward = -0.1
        terminated = False

        # Collision with obstacles
        for obj in self.objects:
            if np.linalg.norm(self.robot_pos - obj["pos"]) < 0.5 + obj["radius"]:
                reward = -10.0
                terminated = True
                break

        # Collision with humans
        for human in self.humans:
            if np.linalg.norm(self.robot_pos - human["pos"]) < 0.5:
                reward = -10.0
                terminated = True
                break

        # Goal reached
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
            (int(self.goal_pos[0] * 50), int(self.goal_pos[1] * 50)),
            15
        )

        # Dessine les obstacles (bleu)
        for obj in self.objects:
            pygame.draw.circle(
                self.screen,
                (0, 0, 255),
                (int(obj["pos"][0] * 50) + 50, int(obj["pos"][1] * 50) + 50),
                int(obj["radius"] * 50)
            )

        # Draw the humans (green + red tip)
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
            
            # Draw the wheelchair (black)
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (int(self.robot_pos[0] * 50) + 25, int(self.robot_pos[1] * 50) + 25, 30, 30)
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
