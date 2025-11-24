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

        # Couleur selon
