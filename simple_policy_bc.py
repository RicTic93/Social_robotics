import numpy as np
import time
from env.fauteuil_env import FauteuilEnv
from config import config
import torch

def generate_expert_demonstrations(env, num_demos, render = False, render_delay = 0.20):  # render_daelay change la vitesse d’affichage des images/parcour
    expert_demonstrations = []
    episode_reward = []

    for _ in range(num_demos):
        obs, _ = env.reset()
        trajectory = []

        done = False

        reward_tot = 0.0

        while not done:
            
            # Calcule l'action à l'aide d'une politique simple
            action = simple_policy(env)

            # Exécute l'action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Enregistre la paire observation-action
            trajectory.append((obs, action))

            obs = next_obs

            reward_tot += reward

            # render --> affichage de l’environnement
            if render:
                env.render()
                time.sleep(render_delay)

        if len(trajectory) > 0:
            expert_demonstrations.append(trajectory)
            episode_reward.append(reward_tot)

    return expert_demonstrations, episode_reward

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
        if dist < 0.8:
            direction += (diff / (dist**2 + 1e-6)) * 2.0
            if dist<env.max_group_distance:
                direction += (diff / (dist**2 + 1e-6)) * 4.0
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

def run_policy(env, model, render=True):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    rewards = []

    while not done:
        # Converti l'osservazione in tensore
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        # Predici l'azione
        with torch.no_grad():
            action = model(obs_tensor).squeeze().numpy()

        # Step nell'ambiente
        next_obs, reward, done, _, info = env.step(action)
        rewards.append(reward)

        total_reward += reward

        if render:
            env.render()

        obs = next_obs

    print("Reward finale:", total_reward)
    for n in rewards:
        if n == -10: print("Hit humans")
        elif n == -7 : print("Hit object")

"""env = FauteuilEnv(config)

# Generate expert demonstrations
num_demos = 10
expert_demonstrations, episode_rewards = generate_expert_demonstrations(env, num_demos)
for i, reward in enumerate(episode_rewards):
    print(f"Episode {i+1}: Total Reward = {reward}")"""