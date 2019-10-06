import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines import PPO2

# if __name__ == "__main__": is necessary on Windows for proper multiprocessing
if __name__ == "__main__":
    # multiprocess environment
    n_cpu = 4
    env = SubprocVecEnv([lambda: gym.make('Self_Driving_Ball:BallGoalGravityEasy-v0') for i in range(n_cpu)])

    model = PPO2(MlpPolicy, env)
    model.learn(total_timesteps=1_000_000)  # 1 mil steps take about 15 minutes... 10 mil steps still appears faulty...
    model.save("PPO2_BGG")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
