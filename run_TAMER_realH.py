import pygame
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle 
import os     
# [MODIFICATION] Import du nouvel environnement RealH
from env.fauteuil_env_realH import FauteuilEnv
from config import config
from tamer.agent_TAMER_realH import Tamer

def plot_proxemic_violations(log_path):
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Fichier {log_path} introuvable.")
        return

    n_episodes = len(df)
    if n_episodes == 0: return
    x = np.arange(n_episodes)
    width = 0.2
    fig, ax = plt.subplots(figsize=(12, 8))

    rects1 = ax.bar(x - width*1.5, df['Intimate Zone Violations'], width, label='Intimate', color='red')
    rects2 = ax.bar(x - width/2, df['Personal Zone Violations'], width, label='Personal', color='orange')
    rects3 = ax.bar(x + width/2, df['Social Zone Violations'], width, label='Social', color='green')

    ax.set_ylabel('Violations')
    ax.set_title('Violations Proxémiques par Épisode (Real H)')
    ax.set_xticks(x)
    ax.set_xticklabels(df['Episode'])
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_metrics(log_path):
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError: return
    if len(df) == 0: return

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(df['Episode'], df['Total Reward'], label='Total Reward'); axs[0, 0].set_title('Récompense Totale'); axs[0, 0].grid(True)
    axs[0, 1].plot(df['Episode'], df['Total Feedback'], label='Total Feedback', color='orange'); axs[0, 1].set_title('Feedback Humain Total'); axs[0, 1].grid(True)
    axs[1, 0].plot(df['Episode'], df['Collisions'], label='Collisions', color='red'); axs[1, 0].set_title('Collisions'); axs[1, 0].grid(True)
    axs[1, 1].plot(df['Episode'], df['Goal Reached'], label='Goal', color='green'); axs[1, 1].set_title('Objectif Atteint (1=Oui)'); axs[1, 1].grid(True)
    plt.tight_layout()
    plt.show()

def visualize_trajectory(episodes_data):
    if not episodes_data:
        print("Aucune donnée de trajectoire à afficher.")
        return

    for i, data in enumerate(episodes_data):
        traj = data['trajectory']
        goal = data['goal_pos']
        objects = data['objects']
        humans = data['humans']
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(traj[:, 0], traj[:, 1], '-o', markersize=2, label='Trajectoire Robot', color='blue')
        ax.scatter(traj[0,0], traj[0,1], c='orange', s=100, label='Départ')
        ax.scatter(goal[0], goal[1], c='green', s=100, marker='*', label='But')
        
        for obj in objects:
            circle = plt.Circle(obj["pos"], obj["radius"], color='black', alpha=0.3)
            ax.add_patch(circle)
            
        for h in humans:
            color = 'purple' if h['type'] == 'dynamic' else 'green'
            ax.scatter(h['pos'][0], h['pos'][1], c=color, s=80, marker='^')

        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f"Trajectoire Épisode {i+1} (Démonstration)")
        plt.show()

def main():
    env = FauteuilEnv(config)

    agent = Tamer(
        env=env,
        num_episodes=5,
        discount_factor=1.0,
        epsilon=0.0,
        min_eps=0.0,
        tame=True,
        ts_len=0.3
    )

    demo_file = 'demonstration_data.p'
    if os.path.exists(demo_file):
        print(f"\nFichier '{demo_file}' trouvé. Chargement...")
        with open(demo_file, 'rb') as f:
            demo_data = pickle.load(f)
        if len(demo_data) > 0:
            agent.pretrain(demo_data)
        else:
            print("Le fichier de démonstration est vide.")
    else:
        print(f"\nAucun fichier '{demo_file}' trouvé. Démarrage sans pré-entraînement.")

    print("\n=== DÉBUT ENTRAÎNEMENT ===")
    env.set_training_mode(True)
    agent.train(model_filename='fauteuil_tamer_model')
    env.set_training_mode(False)

    print("\n=== DÉMONSTRATION PLAY (LIVE) ===")
    
    episodes_data = agent.play(n_episodes=3)

    print("\n=== GÉNÉRATION DES GRAPHIQUES ===")
    plot_metrics(agent.metrics_log_path)
    plot_proxemic_violations(agent.metrics_log_path)
    
    print("\n=== TRAJECTOIRES STATIQUES ===")
    visualize_trajectory(episodes_data)

if __name__ == "__main__":
    main()