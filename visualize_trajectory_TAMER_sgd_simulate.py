import matplotlib.pyplot as plt
import numpy as np
from env.fauteuil_env import FauteuilEnv
from config import config
from tamer.agent_TAMER_sgd_simulate_feedback import Tamer

# Charger l'agent entraîné
env = FauteuilEnv(config)
agent = Tamer(env, num_episodes=2, tame=True)
agent.load_model("example_pretrained_model")  # méthode à implémenter dans Tamer si nécessaire

n_episodes = 2  # ou plus
for ep in range(n_episodes):
    state, _ = agent.env.reset()
    done = False
    trajectory = [state[:2].copy()]
    feedbacks = []

    while not done:
        action = agent.act(state)
        next_state, _, done, _, _ = agent.env.step(action)
        trajectory.append(next_state[:2].copy())
        feedbacks.append(agent._get_human_feedback(next_state))
        state = next_state

    trajectory = np.array(trajectory)
    feedbacks = np.array(feedbacks)

    fig, ax = plt.subplots(figsize=(12, 10))
    for i in range(len(trajectory)-1):
        norm_feedback = (feedbacks[i] + 5) / 10
        color = (1 - norm_feedback, norm_feedback, 0)
        plt.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], color=color, linewidth=2)

    # Humains, obstacles, start, goal
    for human in agent.env.humans:
        ax.scatter(human['pos'][0], human['pos'][1], color='blue', marker='^', s=100, label='Human')
    for obj in agent.env.objects:
        ax.scatter(obj['pos'][0], obj['pos'][1], color='black', marker='o', s=100, label='Obstacle')
    ax.scatter(agent.env.goal_pos[0], agent.env.goal_pos[1], color='green', marker='*', s=200, label='Goal')
    ax.scatter(trajectory[0,0], trajectory[0,1], color='orange', marker='s', s=200, label='Start')

    sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-5, vmax=5))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Feedback Humain')

    ax.set_title(f"Trajectoire Episode {ep+1}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.axis('equal')
    ax.grid(True)
    ax.legend()
    plt.show()
