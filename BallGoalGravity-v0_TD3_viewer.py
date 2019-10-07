import gym

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import TD3


# if __name__ == "__main__": is necessary on Windows for proper multiprocessing
if __name__ == "__main__":
    # TD3 does not allow for multi-processing
    # env = DummyVecEnv([lambda: gym.make('Self_Driving_Ball:BallGoalGravityEasy-v0')])
    # env = DummyVecEnv([lambda: gym.make('Self_Driving_Ball:BallGoalGravityStopEasy-v0')])
    env = DummyVecEnv([lambda: gym.make('Self_Driving_Ball:BallGoalGravity-v0')])  # despite being trained on Easy, the model will solve the regular difficulty!

    model = TD3.load("TD3_BGG")
    # model = TD3.load("TD3_BGGSE")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
