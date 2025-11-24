import pygame
import numpy as np
from env.fauteuil_env import FauteuilEnv
from config import config # Import config dictionnary
from tamer.agent_TAMER_s import Tamer

env = FauteuilEnv(config)
agent = Tamer(env, num_episodes=100, tame=True)
agent.train(model_file_to_save="fauteuil_model")
agent.evaluate(n_episodes=10)
