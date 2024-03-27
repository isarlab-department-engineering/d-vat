import time
import gym
from models.customCNN import CustomCNN
import models.denseMlpPolicy
import wandb
import numpy as np
from stable_baselines3 import SAC
import os
import sys
from pathlib import Path
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from typing import Callable
from UnrealTrackingEnv import UnrealTrackingEnv


def make_env(rank: int,
             seed: int = 0,
             test: bool = False,
             render: bool = False,
             ts: float = 0.1,
             observation_buffer_length: int = 3) -> Callable:

    def _init() -> gym.Env:
        env = UnrealTrackingEnv(rank,
                                test=test,
                                render=render,
                                ts=ts,
                                observation_buffer_length=observation_buffer_length)
        env.seed(seed + rank)
        return Monitor(env)

    set_random_seed(seed + rank)
    return _init


if __name__ == "__main__":

    settings = {'num_cpu': 6,
                'n_timesteps': 10000,
                'batch_size': 64,
                'buffer_size': 10000,
                'train_freq': 8,
                'WandB': False,  # True if you want to use wandb for logging
                'WandB_project': '<your_wandb_project>',
                'WandB_entity': '<your_wandb_entity>',
                'WandB_API_key': '<your_wandb_api_key>',
                'render': True,  # True to disable black screen
                'eval_episodes': 20,
                'eval_mode': True,  # True for model Test - False for Training
                'ts': 0.05,
                'observation_buffer_length': 3
                }

    if settings['eval_mode']:
        # try:
        eval_env = DummyVecEnv([make_env(rank=0,
                                         test=True,
                                         render=settings['render'],
                                         ts=settings['ts'],
                                         observation_buffer_length=settings['observation_buffer_length'])])
        model = SAC.load(os.path.join("experiments/SAC_{}".format(1711119972), "SAC_{}".format(0)), env=eval_env)


        # Eval
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'])
        print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')
    else:
        t = int(time.time())

        # Path for Models
        pathname = os.path.dirname(sys.argv[0])
        abs_path = os.path.abspath(pathname)
        current_path = Path(os.path.join(abs_path, "experiments", "SAC_{}".format(t)))
        current_path.mkdir(parents=True, exist_ok=True)

        if settings['WandB']:
            wandb.login(key=settings['WandB_API_key'])
            wandb.init(project=settings['WandB_project'], entity=settings['WandB_entity'],
                       name="SAC_{}".format(t), config=settings)

        # Multiprocess RL Training
        vec_env = SubprocVecEnv([make_env(rank=i + 1,
                                          test=False,
                                          render=settings['render'],
                                          ts=settings['ts'],
                                          observation_buffer_length=settings['observation_buffer_length'])
                                 for i in range(settings['num_cpu'])])

        policy_kwargs = dict(
            net_arch=[512]
        )

        model = SAC('DenseMlpPolicy',
                    vec_env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    buffer_size=settings['buffer_size'],
                    batch_size=settings['batch_size'],
                    train_freq=settings['train_freq'])

        print(model.policy)
        print(model.critic)

        # Separate environment for evaluation
        eval_env = DummyVecEnv([make_env(rank=0,
                                         test=True,
                                         render=settings['render'],
                                         ts=settings['ts'],
                                         observation_buffer_length=settings['observation_buffer_length'])])
        best_episodes = np.zeros((10,))

        # Training
        while True:
            model.learn(settings['n_timesteps'])

            # Eval
            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=settings['eval_episodes'], render=settings['render'])
            print(f'Mean reward: {mean_reward} +/- {std_reward:.2f}')

            if settings['WandB']:
                wandb.log({'test': mean_reward})

            worst_model = np.argmin(best_episodes)
            if mean_reward > best_episodes[worst_model]:
                best_episodes[worst_model] = mean_reward
                model.save(os.path.join(current_path, "SAC_{}".format(worst_model)))

