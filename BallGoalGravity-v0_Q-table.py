import numpy as np
import gym

env = gym.make('Self_Driving_Ball:BallGoalGravity-v0')

HM_EPISODES = 10000  # how many episodes
epsilon = 1
START_EPSILON_DECAYING = 0  # which episode to start decaying
END_EPSILON_DECAYING = HM_EPISODES // 2
EPSILON_DECAY_VALUE = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
SHOW_EVERY = 1
INITIAL_REWARD_LOWER_BOUND = -10

LEARNING_RATE = 0.1
DISCOUNT = 0.95


DIS_NUMBER_OBS = 20  # how many discrete segments for observation space
DIS_NUMBER_ACT = 4  # how many discrete segments for action space
DISCRETE_OS_SIZE = [DIS_NUMBER_OBS + 1] * len(env.observation_space.high)  # e.g. [20 + 1, 20 + 1, 20 + 1, 20 + 1]; +1 because for x segments, there exists x+1 action choices
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / ([DIS_NUMBER_OBS] * len(env.observation_space.high))
DISCRETE_AC_SIZE = [DIS_NUMBER_ACT + 1] * len(env.action_space.high)  # e.g. [4 + 1, 4 + 1]
dax_size, day_size = (env.action_space.high - env.action_space.low) / ([DIS_NUMBER_ACT] * len(env.action_space.high))
ax_min, ay_min = env.action_space.low


q_table = np.random.uniform(low=INITIAL_REWARD_LOWER_BOUND, high=0, size=(DISCRETE_OS_SIZE + DISCRETE_AC_SIZE))

ep_rewards = []


def get_discrete_state(continuous_state):
    output_state = (continuous_state - env.observation_space.low) / discrete_os_win_size
    return tuple(output_state.astype(np.int))


# return the best discrete action by looking up a discrete observation in the action space part of the q_table
def get_best_discrete_action(action_space):
    current_max = -float('inf')
    best_dax = 0
    best_day = 0
    for dax in range(DIS_NUMBER_ACT + 1):
        for day in range(DIS_NUMBER_ACT + 1):
            if action_space[dax][day] > current_max:
                current_max = action_space[dax][day]
                best_dax = dax
                best_day = day
    return best_dax, best_day, current_max


for episode in range(HM_EPISODES):
    episode_reward = 0
    discrete_state = get_discrete_state(env.reset())
    if not episode % SHOW_EVERY:  # Python quirk; equivalent to "if A % B == 0"
        render = True
    else:
        render = False

    done = False
    while not done:
        if render:
            env.render()

        if np.random.random() > epsilon:
            dax, day, current_q = get_best_discrete_action(q_table[discrete_state])
        else:
            dax, day = np.random.randint(0, DIS_NUMBER_ACT + 1, size=2)
            current_q = q_table[discrete_state][dax][day]

        new_state, reward, done, _ = env.step(np.asarray(a=(ax_min + dax * dax_size, ay_min + day * day_size)))
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if not done:
            _, _, max_future_q = get_best_discrete_action(q_table[new_discrete_state])  # could be made more efficient, by fetching the next best action here...
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state][dax][day] = new_q
            discrete_state = new_discrete_state
        elif reward > 0:  # assuming that reaching the goal is the only way to yield a positive reward
            print(f'we made it on episode {episode}')
            q_table[discrete_state][dax][day] = reward

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= EPSILON_DECAY_VALUE

    ep_rewards.append(episode_reward)

    # yes... this maps the first ever episode
    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])

        print(f'Episode: {episode}, average reward: {average_reward}, min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}')

env.close()


