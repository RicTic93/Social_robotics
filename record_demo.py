import pygame
import numpy as np
import pickle
import os
from env.fauteuil_env_realH import FauteuilEnv 
from config import config

ACTIONS = [
    np.array([0.0, 0.0]),   # 0: Stop
    np.array([0.0, -0.5]),  # 1: Avant
    np.array([0.0, 0.5]),   # 2: Arrière
    np.array([-0.5, 0.0]),  # 3: Gauche
    np.array([0.5, 0.0])    # 4: Droite
]

def get_action_from_key():
    keys = pygame.key.get_pressed()

    if keys[pygame.K_UP] or keys[pygame.K_z]: return 1
    if keys[pygame.K_DOWN] or keys[pygame.K_s]: return 2
    if keys[pygame.K_LEFT] or keys[pygame.K_q]: return 3
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]: return 4
    return 0 # Stop

def main():
    env = FauteuilEnv(config)
    obs, _ = env.reset()
    
    demonstration_data = [] # 存储 [(state, action_idx), ...]
    running = True
    clock = pygame.time.Clock()

    print("=================================================")
    print("MODE ENREGISTREMENT (EXPERT DEMO)")
    print("Contrôlez le robot avec les flèches pour atteindre le but.")
    print("Appuyez sur 'E' pour sauvegarder et quitter.")
    print("=================================================")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_e:
                running = False

        action_idx = get_action_from_key()
        continuous_action = ACTIONS[action_idx]

        if action_idx != 0:
            demonstration_data.append((obs, action_idx))

        next_obs, reward, terminated, truncated, _ = env.step(continuous_action)
        
        env.render()

        if terminated:
            print("But atteint ! (Fin de l'épisode)")
            running = False

        obs = next_obs
        clock.tick(30)

    env.close()

    # Save data
    if len(demonstration_data) > 0:
        save_path = 'demonstration_data.p'
        with open(save_path, 'wb') as f:
            pickle.dump(demonstration_data, f)
        print(f"\nDémonstration sauvegardée : {len(demonstration_data)} pas dans '{save_path}'")
    else:
        print("\nAucune donnée enregistrée.")

if __name__ == "__main__":
    main()