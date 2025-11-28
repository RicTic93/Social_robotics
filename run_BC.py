import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import time
import numpy as np
import matplotlib.pyplot as plt

from env.fauteuil_env import FauteuilEnv
from config import config 
from BC.simple_policy_bc import generate_expert_demonstrations
from BC.simple_policy_bc import run_policy
from BC.bc_model import BehavioralCloningModel

states = []
actions = []

env = FauteuilEnv(config)
num_demos = 250 # Generate expert demonstrations
expert_demonstrations, episode_rewards = generate_expert_demonstrations(env, num_demos)
print("Generate expert demonstrations, collecting data and training BC agent...")

for demonstration in expert_demonstrations:
    for state, action in demonstration:
        states.append(state)
        actions.append(action)

states = np.array(states, dtype=np.float32)
actions = np.array(actions, dtype=np.float32)

# Define model (combien dimensions d'état et d'action)
state_dim = states.shape[1]
action_dim = actions.shape[1]

agent = BehavioralCloningModel(state_dim, action_dim)

# Train the BC agent
loss_values, loss_std = agent.train(states, actions, epochs=50, batch_size=32)

# Teste le modèle entraîné
total_reward_final, trajectory_length_final, completed_final, hit_humans_final, hit_objects_final = run_policy(env, agent, render=True)
time.sleep(0.15)

print(f"Total Reward after BC: {total_reward_final}")
print(f"Trajectory Length after BC: {trajectory_length_final}")
print(f"Completed after BC: {completed_final}")
print(f"Human Hits after BC: {hit_humans_final}")
print(f"Object Hits after BC: {hit_objects_final}")

# ------ Plotting results ------
epochs = 50
lighter_blue = "b"

# Diminution erreur pendant l'entraînement (loss)
plt.figure(figsize=(12, 5))

plt.plot(range(1, epochs + 1), loss_values, color='b', label='Loss')
plt.fill_between(range(1, epochs + 1), loss_values - loss_std, loss_values + loss_std,
                 color=lighter_blue, alpha=0.3, label='Loss ± std')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)

# Compareson des longueurs de trajectoires expertes et BC
trajectory_lengths = [len(traj) for traj in expert_demonstrations]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(trajectory_lengths, color='b', label='Expert Trajectory Lengths')
plt.scatter([len(trajectory_lengths)], [trajectory_length_final], color='r', s=100,
            label='BC Final Trajectory Length', zorder=5)
plt.xlabel('Episode')
plt.ylabel('Trajectory Length')
plt.title('Expert vs BC Trajectory Lengths')
plt.legend()
plt.grid(True)
plt.yticks(np.arange(0, max(trajectory_lengths + [trajectory_length_final]) + 10, 10))

# Compareson des récompenses des épisodes expertes et BC
plt.subplot(1,2,2)
plt.plot(episode_rewards, color='b', label='Episode reward')
plt.scatter([len(episode_rewards)], [total_reward_final], color='r', s=100,
            label='Reward after BC', zorder=5)
plt.xlabel('Episode')
plt.ylabel('Reward for episode')
plt.title('Expert vs BC Reward')
plt.legend()
plt.grid(True)
plt.yticks(np.arange(0, max(episode_rewards + [total_reward_final]) + 10, 10))

plt.tight_layout()
plt.show()