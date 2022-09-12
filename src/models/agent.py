from sb3_contrib.ppo_mask import MaskablePPO


def create_agent(algorithm="MaskablePPO", policy="MultiInputPolicy", policy_kwargs=None, env=None, log_dir=None, verbose=1):
    if algorithm == "MaskablePPO":
        #stopTrainingOnMaxEpisodes_callback = StopTrainingOnMaxEpisodes(max_episodes = n_episodes, verbose=verbose)
        #tensorboard_callback = TensorboardCallback()
        #saveOnBestTrainingReward_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_dir=models_dir, verbose=verbose)
        '''
        eval_callback = EvalCallback(env, best_model_save_path='models/jss/PPO/best_model',
                             log_path=log_dir, eval_freq=5,
                             deterministic=False, render=False)
        '''
        # Create the callback list
        #callback = CallbackList([stopTrainingOnMaxEpisodes_callback, saveOnBestTrainingReward_callback, tensorboard_callback])

        model = MaskablePPO(
            policy='MultiInputPolicy', # alias of MaskableMultiInputActorCriticPolicy
            env=env, 
            policy_kwargs=policy_kwargs,
            verbose=verbose, 
            tensorboard_log=log_dir)
        #model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir)
        return model