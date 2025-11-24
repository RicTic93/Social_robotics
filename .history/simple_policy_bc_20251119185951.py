import numpy as np
import time
from env.fauteuil_env import FauteuilEnv
from config import config

def generate_expert_demonstrations(env, num_demos, render = True, render_delay = 0.20):  # render_daelay change la vitesse d’affichage des images/parcour
    expert_demonstrations = []

    for _ in range(num_demos):
        obs, _ = env.reset()
        trajectory = []

        done = False

        while not done:
            # Calcule l'action à l'aide d'une politique simple
            action = simple_policy(env)

            # Exécute l'action
            next_obs, reward, done, bool, info = env.step(action)

            # Enregistre la paire observation-action
            trajectory.append((obs, action))

            obs = next_obs

            # render --> affichage de l’environnement
            if render:
                env.render()
                time.sleep(render_delay)

        if len(trajectory) > 0:
            expert_demonstrations.append(trajectory)

    return expert_demonstrations

def simple_policy(env):
    robot = env.robot_pos
    goal = env.goal_pos

    # Simple policy
    direction = goal - robot
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm  # normalise pour rester dans [-1,1]

    # Évite les objets
    for obj in env.objects:
        diff = robot - obj["pos"]
        dist = np.linalg.norm(diff)
        if dist < obj["radius"] + 1.0:
            direction += (diff / (dist**2 + 1e-6)) * 2.0
        """op = obj["pos"]
        r  = obj["radius"]
        dist = np.linalg.norm(robot - op)

        if dist < r + 0.4:   # se troppo vicino, scappa all’indietro
            avoid = robot - op
            avoid /= np.linalg.norm(avoid)
            direction = avoid    # priotità all avoidance"""

    # Évite les humains
    for human in env.humans:
        diff = robot - human["pos"]
        dist = np.linalg.norm(diff)
        if dist < 0.6:
            direction += (diff / (dist**2 + 1e-6)) * 2.0
        """hp = human["pos"]
        dist = np.linalg.norm(robot - hp)

        if dist < 0.6:
            avoid = robot - hp
            avoid /= np.linalg.norm(avoid)
            direction = avoid"""

    # Normalise l'action pour respecter l'espace d'action [-1,1]
    direction += np.random.uniform(-0.6, 0.6, size=2)
    action = direction
    norm = np.linalg.norm(action)
    if norm > 1:
        action = action / norm

    return action.astype(np.float32)

env = FauteuilEnv(config)

# Generate expert demonstrations
num_demos = 10 
expert_demonstrations = generate_expert_demonstrations(env, num_demos)
