import numpy as np
import time
import torch

def generate_expert_demonstrations(env, num_demos, render = False, render_delay = 0.20):  # render_daelay change la vitesse d’affichage des images/parcour
    expert_demonstrations = []
    episode_reward = []

    for _ in range(num_demos):
        obs, _ = env.reset()
        trajectory = []
        done = False
        reward_tot = 0.0

        # L'action choisie par les experts est exécutée dans l'environnement
        while not done:
            action = simple_policy(env)
            next_obs, reward, terminated, truncated, info = env.step(action) 
            done = terminated or truncated
            trajectory.append((obs, action))
            obs = next_obs

            reward_tot += reward

            if render:  # Affiche environnement
                env.render()
                time.sleep(render_delay)
    
        if len(trajectory) > 0:
            expert_demonstrations.append(trajectory)
            episode_reward.append(reward_tot)

    return expert_demonstrations, episode_reward

# Policy pour les démonstrations expertes
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
    # Évite les humains
    for human in env.humans:
        diff = robot - human["pos"]
        dist = np.linalg.norm(diff)
        if dist < 0.8:
            direction += (diff / (dist**2 + 1e-6)) * 2.0
            if dist<1.2:
                direction += (diff / (dist**2 + 1e-6)) * 4.0

    # Normalise l'action pour respecter l'espace d'action [-1,1]
    direction += np.random.uniform(-0.6, 0.6, size=2)
    action = direction
    norm = np.linalg.norm(action)
    if norm > 1:
        action = action / norm

    return action.astype(np.float32) # return l'actoin choise par les expert

# Exécute la politique apprise dans l'environnement
def run_policy(env, model, render):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    rewards = [] # Pour valutation du tipe de récompense cumulé
    trajectory_length = 0 
    hit_humans = 0 # Track le nombre de collisions
    hit_objects = 0

    while not done:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        # Prediction de l'action
        with torch.no_grad():
            action = model(obs_tensor).squeeze().numpy()
        next_obs, reward, done, _, info = env.step(action)

        rewards.append(reward)
        total_reward += reward
        trajectory_length += 1

        if trajectory_length == 500:
            return total_reward, trajectory_length, False, hit_humans, hit_objects

        if render:
            env.render()

        obs = next_obs

    for n in rewards:
        if n == -10: hit_humans += 1
        elif n == -7 : hit_objects += 1

    return total_reward, trajectory_length, True, hit_humans, hit_objects