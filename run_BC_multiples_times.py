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

num_experiments = 10
results = []

print(f"Starting a set of {num_experiments} experiments...\n")

for exp_num in range(num_experiments):
    states = []
    actions = []

    env = FauteuilEnv(config)
    num_demos = 250 # Generate expert demonstrations
    expert_demonstrations, episode_rewards = generate_expert_demonstrations(env, num_demos)

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
    loss_values, loss_std = agent.train(states, actions, epochs=50, batch_size=50)

    # Teste le modèle entraîné
    total_reward_final, trajectory_length_final, completed, hit_humans_final, hit_objects_final = run_policy(env, agent, render=True)
    time.sleep(0.15)

    results.append({
        "total_reward": total_reward_final,
        "mean_episode_reward": np.mean(episode_rewards),
        "trajectory_length": trajectory_length_final,
        "mean_trajectory_length": np.mean([len(traj) for traj in expert_demonstrations]),
        "completed": completed,
        "Human_hits": hit_humans_final,
        "Object_hits": hit_objects_final
    })

    print(f"--- Experiment {exp_num + 1} completed ---\n")

# Affiche les résultats
for i, res in enumerate(results):
    print(f"Experiment {i+1}:")
    print(f"Total Reward = {res['total_reward']} / Mean Expert Reward = {res['mean_episode_reward']}")
    print(f"Trajectory Length = {res['trajectory_length']} / Mean Expert Trajectory Length = {res['mean_trajectory_length']}")
    print(f"Completed = {res['completed']}")
    print(f"Human Hits = {res['Human_hits']}, Object Hits = {res['Object_hits']}")
    
# ------ Plotting results ------
plt.style.use('_mpl-gallery')
plt.figure(figsize=(12, 5))

ax = plt.subplot(1, 2, 1)
# Comparaison des longueurs de trajectoires expertes et BC
x = 1 +np.arange(len(results))
y = [res['trajectory_length'] for res in results]
y2 = [res['mean_trajectory_length'] for res in results]

ax.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='k-', label="Trajectory Length")
ax.plot(x, y2, 'r-', linewidth=2, label="Mean Expert Trajectory Length")
ax.set(
    xlim=(0, len(results)+1),
    xticks=np.arange(1, len(results)+1),
)
ax.set_title("Trajectory vs Expert Mean")
ax.set_xlabel("Number of Experiment")
ax.set_ylabel("Length")
ax.legend()

ax = plt.subplot(1, 2, 2)
# Comparaison des récompenses des épisodes expertes et BC
x = 1 +np.arange(len(results))
y = [res['total_reward'] for res in results]
y2 = [res['mean_episode_reward'] for res in results]

ax.stem(x, y, linefmt='b-', markerfmt='bo', basefmt='k-', label="Reward Total")
ax.plot(x, y2, 'r-', linewidth=2, label="Mean Expert Reward")
ax.set(
    xlim=(0, len(results)+1),
    xticks=np.arange(1, len(results)+1),
)
ax.set_title("Reward vs Expert Mean")
ax.set_xlabel("Number of Experiment")
ax.set_ylabel("Reward")
ax.legend()

plt.tight_layout()
plt.show()