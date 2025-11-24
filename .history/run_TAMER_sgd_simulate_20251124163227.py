import pygame
import numpy as np
from env.fauteuil_env import FauteuilEnv
from config import config  # Import config dictionnary
from tamer.agent_TAMER_simulate_feedback import Tamer
import matplotlib.pyplot as plt  # <- Pour la visualisation

# --- Ton code existant ---
env = FauteuilEnv(config)
agent = Tamer(env, num_episodes=5, tame=True)
agent.train(model_file_to_save="fauteuil_model")
agent.evaluate(n_episodes=5)

# --- Visualisation de la trajectoire ---
def visualize_trajectory(agent, n_episodes=1):
    for ep in range(n_episodes):
        state, _ = agent.env.reset()
        done = False
        trajectory = [state[:2].copy()]
        feedbacks = []
        while not done:
            action = agent.act(state)
            next_state, _, done, _, info = agent.env.step(action)
            trajectory.append(next_state[:2].copy())
            # Récupère le feedback humain pour colorer la trajectoire
            feedbacks.append(info.get("human_feedback", 0))
            state = next_state
        trajectory = np.array(trajectory)
        feedbacks = np.array(feedbacks)

        # Couleur selon le feedback (vert = positif, rouge = négatif)
        colors = np.where(feedbacks >= 0, 'green', 'red')
        for i in range(len(trajectory)-1):
            plt.plot(trajectory[i:i+2,0], trajectory[i:i+2,1], color=colors[i])

    # Affiche les humains
    for human in agent.env.humans:
        plt.scatter(human['pos'][0], human['pos'][1], color='blue', marker='x', s=100, label='Human')

    plt.title("Trajectoire du robot avec feedback humain")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.show()

# Appel de la fonction
visualize_trajectory(agent, n_episodes=3)
