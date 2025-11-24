import pygame
import numpy as np
from env.fauteuil_env import FauteuilEnv
from config import config  # Import config dictionnary

def read_keyboard():
    keys = pygame.key.get_pressed()
    current_action = np.array([0.0, 0.0])

    if keys[pygame.K_z]:
        current_action[1] = -0.5
    if keys[pygame.K_s]:
        current_action[1] = 0.5
    if keys[pygame.K_q]:
        current_action[0] = -0.5
    if keys[pygame.K_d]:
        current_action[0] = 0.5

    # Diagonales
    if keys[pygame.K_z] and keys[pygame.K_q]:
        current_action = np.array([-0.5, -0.5])
    if keys[pygame.K_z] and keys[pygame.K_d]:
        current_action = np.array([0.5, -0.5])
    if keys[pygame.K_s] and keys[pygame.K_q]:
        current_action = np.array([-0.5, 0.5])
    if keys[pygame.K_s] and keys[pygame.K_d]:
        current_action = np.array([0.5, 0.5])

    return current_action

def main():
    pygame.init()
    env = FauteuilEnv(config)  # Passe le dictionnaire config à FauteuilEnv
    obs, _ = env.reset()
    current_action = np.array([0.0, 0.0])
    running = True
    clock = pygame.time.Clock()

    print("🚀 Environnement initialisé. Utilise ZQSD pour contrôler le fauteuil.")
    
    # Dans la boucle principale (main), ajoute :
    print(f"Humains dans FOV: {[env._is_in_field_of_view(env.robot_pos, env.goal_pos, h['pos']) for h in env.humans]}")


    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_action = read_keyboard()
        obs, reward, terminated, truncated, info = env.step(current_action)
        print(f"Position : {env.robot_pos} | Récompense : {reward:.2f}")

        env.render()

        if terminated:
            print("🎯 But atteint ou collision ! Réinitialisation...")
            obs, _ = env.reset()

        clock.tick(30)

    env.close()
    print("Programme terminé.")

if __name__ == "__main__":
    main()
