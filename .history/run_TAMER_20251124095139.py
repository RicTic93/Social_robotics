from fauteuil_env i

env = FauteuilEnv(config)
agent = Tamer(env, num_episodes=100, tame=True)
agent.train(model_file_to_save="fauteuil_model")
agent.evaluate(n_episodes=10)
