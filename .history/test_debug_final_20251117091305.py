import pipserial
import threading
import numpy as np
import pygame
import time
from env.fauteuil_env import FauteuilEnv
from config import config

current_action = np.array([0.0, 0.0])
running = True
use_joystick = True

def read_arduino():
    global current_action, running, use_joystick
    try:
        arduino = serial.Serial('COM4', 9600, timeout=0.1)
        buffer = ""
        deadzone = 0.1
        print("🔌 Connexion Arduino établie sur COM4. Bouge le joystick !")
        while running:
            data = arduino.read(arduino.in_waiting or 1).decode('utf-8')
            if data:
                buffer += data
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    try:
                        x, y = map(float, line.strip().split(','))
                        x = x if abs(x) > deadzone else 0.0
                        y = y if abs(y) > deadzone else 0.0
                        current_action = np.array([x, y])
                        print(f"🕹️ Joystick : X={x:.2f}, Y={y:.2f}")
                    except ValueError:
                        print(f"⚠️ Données invalides : {line.strip()}")
            time.sleep(0.01)
    except serial.SerialException:
        print("⚠️ Joystick non détecté. Utilisation du clavier.")
        use_joystick = False
    except Exception as e:
        print(f"❌ Erreur Arduino : {e}")
        use_joystick = False

def read_keyboard():
    global current_action, running
    keys = pygame.key.get_pressed()
    current_action = np.array([0.0, 0.0])

    if keys[pygame.K_z]:
        current_action[1] = -1.0
    if keys[pygame.K_s]:
        current_action[1] = 1.0
    if keys[pygame.K_q]:
        current_action[0] = -1.0
    if keys[pygame.K_d]:
        current_action[0] = 1.0

    if keys[pygame.K_z] and keys[pygame.K_q]:
        current_action = np.array([-0.7, -0.7])
    if keys[pygame.K_z] and keys[pygame.K_d]:
        current_action = np.array([0.7, -0.7])
    if keys[pygame.K_s] and keys[pygame.K_q]:
        current_action = np.array([-0.7, 0.7])
    if keys[pygame.K_s] and keys[pygame.K_d]:
        current_action = np.array([0.7, 0.7])

    print(f"🎮 Clavier : {current_action}")

arduino_thread = threading.Thread(target=read_arduino)
arduino_thread.daemon = True
arduino_thread.start()

env = FauteuilEnv(config)
obs, _ = env.reset()
print("🚀 Environnement initialisé. Position initiale :", env.robot_pos)

try:
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("🚪 Fenêtre fermée par l'utilisateur.")
                running = False
                break

        if not use_joystick:
            read_keyboard()

        obs, reward, terminated, truncated, info = env.step(current_action)
        print(f"📌 Position : {env.robot_pos} | Récompense : {reward:.2f}")

        env.render()

        if terminated:
            print("🎯 But atteint ! Réinitialisation...")
            obs, _ = env.reset()

except KeyboardInterrupt:
    print("\n⏹️ Arrêt demandé par l'utilisateur.")
finally:
    running = False
    env.close()
    print("🧹 Nettoyage terminé.")
