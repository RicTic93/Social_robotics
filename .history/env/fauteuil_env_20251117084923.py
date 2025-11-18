import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class FauteuilEnv(gym.Env):
    def __init__(self, config):
        super(FauteuilEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Génère un nombre total d'éléments entre 4 et 20
        total_elements = np.random.randint(4, 21)

        # Répartit équitablement entre humains et obstacles
        num_humans = total_elements // 2
        num_objects = total_elements - num_humans

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4 + 3*num_objects + 2*num_humans,), dtype=np.float32)

        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)
        self.goal_pos = np.array([8.0, 8.0], dtype=np.float32)
        self.max_speed = 0.5

        # Génère les obstacles en évitant les chevauchements
        self.objects = []
        min_distance = 1.5  # Distance minimale entre les obstacles
        for _ in range(num_objects):
            valid = False
            attempts = 0
            while not valid and attempts < 100:
                pos = np.random.uniform(0, 10, size=2)
                radius = np.random.uniform(0.3, 0.8)
                valid = True
                for obj in self.objects:
                    if np.linalg.norm(pos - obj["pos"]) < min_distance + obj["radius"] + radius:
                        valid = False
                        break
                if valid:
                    self.objects.append({"pos": pos, "radius": radius})
                attempts += 1

        # Génère les humains
        self.humans = []
        for _ in range(num_humans):
            self.humans.append({
                "pos": np.random.uniform(0, 10, size=2),
                "direction": np.random.uniform(-1, 1, size=2),
                "duration": np.random.randint(30, 60)
            })

        pygame.init()
        self.screen = pygame.display.set_mode((600, 600))
        pygame.display.set_caption("Fauteuil Roulant Intelligent")
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        self.robot_pos = np.array([1.0, 1.0], dtype=np.float32)

        # Réinitialise les positions des humains
        for human in self.humans:
            human["pos"] = np.random.uniform(0, 10, size=2)
            human["direction"] = np.random.uniform(-1, 1, size=2)
            norm = np.linalg.norm(human["direction"])
            if norm > 0:
                human["direction"] = human["direction"] / norm
            human["duration"] = np.random.randint(30, 60)

        return self._get_obs(), {}

    def move_humans(self):
        for i, human in enumerate(self.humans):
            # Si la durée est écoulée, choisis une nouvelle direction
            if human["duration"] <= 0:
                human["direction"] = np.random.uniform(-1, 1, size=2)
                norm = np.linalg.norm(human["direction"])
                if norm > 0:
                    human["direction"] = human["direction"] / norm
                human["duration"] = np.random.randint(30, 60)

            # Déplace la personne dans la direction actuelle
            new_pos = human["pos"] + human["direction"] * 0.05
            human["duration"] -= 1

            # Vérifie que la nouvelle position reste dans les limites de l'écran
            new_pos = np.clip(new_pos, 0, 10)

            # Vérifie la distance avec les autres personnes
            valid = True
            for j, other_human in enumerate(self.humans):
                if i != j and np.linalg.norm(new_pos - other_human["pos"]) < 1.0:
                    valid = False
                    break

            # Vérifie la distance avec les obstacles
            if valid:
                for obj in self.objects:
                    if np.linalg.norm(new_pos - obj["pos"]) < 0.5 + obj["radius"]:
                        valid = False
                        break

            # Si la position est valide, met à jour
            if valid:
                self.humans[i]["pos"] = new_pos

    def step(self, action):
        self.robot_pos += action * self.max_speed
        self.robot_pos = np.clip(self.robot_pos, 0, 10)

        # Déplace les personnes
        #self.move_humans()

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

        # Dessine les personnes (rouge)
        for human in self.humans:
            pygame.draw.circle(
                self.screen,
                (255, 0, 0),
                (int(human["pos"][0] * 50) + 50, int(human["pos"][1] * 50) + 50),
                20
            )

        # Dessine le fauteuil (noir)
        pygame.draw.rect(
            self.screen,
            (0, 0, 0),
            (int(self.robot_pos[0] * 50) + 25, int(self.robot_pos[1] * 50) + 25, 50, 50)
        )

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.quit()
