import gym
import numpy as np

from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3
from stable_baselines.ddpg.noise import NormalActionNoise


# if __name__ == "__main__": is necessary on Windows for proper multiprocessing
if __name__ == "__main__":
    # TD3 does not allow for multi-processing
    env = DummyVecEnv([lambda: gym.make('Self_Driving_Ball:BallGoalGravityEasy-v0')])  # agent trained on Easy can solve the regular version of the environment

    # The noise objects for TD3
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = TD3(MlpPolicy, env, action_noise=action_noise)
    model.learn(total_timesteps=100_000)  # at 100_000 episodes, it seems to be working at 100%! Takes about 15 minutes.
    model.save("TD3_BGGE")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
