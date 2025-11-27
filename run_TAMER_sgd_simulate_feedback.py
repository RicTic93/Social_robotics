import os
from pathlib import Path
# Permet d'éviter les crashes liés à OpenMP / Mac Intel
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from datetime import datetime
from env.fauteuil_env import FauteuilEnv
from config import config
from tamer.agent_TAMER_sgd_simulate_feedback import Tamer

# Dossier logs dans Tamer
LOGS_DIR = Path(__file__).parent.joinpath("tamer/logs")
LOGS_DIR.mkdir(exist_ok=True, parents=True) 
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# --- Initialisation ---
env = FauteuilEnv(config)
agent = Tamer(env, num_episodes=2, tame=True)

# --- Entraînement ---
agent.train(model_file_to_save="fauteuil_model")

# --- Évaluation ---
agent.evaluate(n_episodes=1)

# --- Sauvegarde des métriques dans un CSV ---
metrics = {
    'Episode': list(range(1, len(agent.losses)+1)),
    'Loss': agent.losses,
    'Total Reward': agent.total_rewards,
    'Total Feedback': agent.total_feedbacks,
    'Intimate Zone Violations': agent.intimate_violations_per_episode,
    'Personal Zone Violations': agent.personal_violations_per_episode,
    'Social Zone Violations': agent.social_violations_per_episode,
    'Public Zone Violations': agent.public_violations_per_episode,
    'Collisions': agent.collisions_per_episode,
    'Goal Reached': agent.goals_per_episode,
    'Predictability': agent.predictabilities,
    'Politeness': agent.politenesses
}

df = pd.DataFrame(metrics)
log_file = LOGS_DIR.joinpath(f"metrics_{current_time}.csv")
df.to_csv(log_file, index=False)
print(f"Metrics saved to {log_file}")