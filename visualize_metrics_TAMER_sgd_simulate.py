import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

log_file = "tamer/logs/metrics_log150.csv"
df = pd.read_csv(log_file)

# --- Loss lissée ---
def moving_average(x, window=5):
    return np.convolve(x, np.ones(window)/window, mode='valid')

# --- Affichage de la Loss lissée ---
loss = df['Loss'].values
smoothed_loss = moving_average(loss, window=5)

plt.figure(figsize=(10,6))
plt.plot(loss, alpha=0.3, label="Loss brute")
plt.plot(smoothed_loss, linewidth=2, label="Loss lissée (MA=5)")
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.title('Loss lissée')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.grid(True)
plt.show()

# --- Total Reward par Episode ---
plt.figure(figsize=(10,6))
plt.plot(df['Episode'], df['Total Reward'], label='Total Reward', color='green')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Total Reward par Episode')
# → Demande à Matplotlib de ne garder que quelques ticks
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.grid(True)
plt.legend()
plt.show()

# --- Violations proxémiques ---
plt.figure(figsize=(10,6))
plt.plot(df['Episode'], df['Intimate Zone Violations'], label='Intimate', color='red')
plt.plot(df['Episode'], df['Personal Zone Violations'], label='Personal', color='orange')
plt.plot(df['Episode'], df['Social Zone Violations'], label='Social', color='green')
plt.plot(df['Episode'], df['Public Zone Violations'], label='Public', color='blue')
plt.xlabel('Episode')
plt.ylabel('Violations')
plt.title('Proxemic Zone Violations')
# → Demande à Matplotlib de ne garder que quelques ticks
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.grid(True)
plt.legend()
plt.show()

# --- Predictabilité par Episode ---
plt.figure(figsize=(10,6))
plt.plot(df['Episode'], df['Predictability'], label='Predictability (P3)', color='purple')
plt.xlabel('Episode')
plt.ylabel('Predictability Score')
plt.title('Lisibilité de trajectoire (P3) par Épisode')
plt.xticks(df['Episode'][::50])
plt.grid(True)
plt.legend()
plt.show()

# --- Politesse par Episode ---
plt.figure(figsize=(10,6))
plt.plot(df['Episode'], df['Politeness'], label='Politeness (P4)', color='brown')
plt.xlabel('Episode')
plt.ylabel('Politeness Score')
plt.title('Politesse (P4) par Épisode')
plt.xticks(df['Episode'][::50])
plt.grid(True)
plt.legend()
plt.show()

# --- Graphique des collisions par épisode ---
plt.figure(figsize=(10,6))
plt.bar(df['Episode'], df['Collisions'], color='red', alpha=0.7)
plt.xlabel('Episode')
plt.ylabel('Nombre de collisions')
plt.title('Nombre de collisions par épisode')
# → Demande à Matplotlib de ne garder que quelques ticks
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.grid(axis='y')
plt.show()

# --- Graphique combiné avec Reward et Loss ---
fig, ax1 = plt.subplots(figsize=(12,6))
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward', color='green')
ax1.plot(df['Episode'], df['Total Reward'], color='green', label='Total Reward')
ax1.tick_params(axis='y', labelcolor='green')
plt.xticks(df['Episode'][::50])
ax1.grid(True)

ax2 = ax1.twinx()  # second axe pour Loss
ax2.set_ylabel('Loss', color='red')
ax2.plot(df['Episode'], df['Loss'], color='red', label='Loss')
ax2.tick_params(axis='y', labelcolor='red')
plt.locator_params(axis='x', nbins=10)
plt.locator_params(axis='y', nbins=6)
plt.title("Reward et Loss par épisode")
fig.tight_layout()
plt.show()