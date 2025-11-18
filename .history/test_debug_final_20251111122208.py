import serial
import threading
import numpy as np
import pygame
import time
from env.fauteuil_env import FauteuilEnv
from config import config

current_action = np.array([0.0, 0.0])
running = True

def read_arduino():
    global current_action, running
    arduino = serial.Serial('COM4', 9600, timeout=0.1) #com4 a changer selon le pc
    buffer = ""
    deadzone = 0.1  # Zone morte pour ignorer les petites valeurs
    print("🔌 Connexion Arduino établie sur COM4. Bouge le joystick !")
    while running:
        data = arduino.read(arduino.in_waiting or 1).decode('utf-8')
        if data:
            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                try:
                    x, y = map(float, line.strip().split(','))
                    # Applique la zone morte
                    x = x if abs(x) > deadzone else 0.0
                    y = y if abs(y) > deadzone else 0.0
                    current_action = np.array([x, y])
                    print(f"🕹️ Joystick : X={x:.2f}, Y={y:.2f}")
                except ValueError:
                    print(f"⚠️ Données invalides : {line.strip()}")
        time.sleep(0.01)

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
