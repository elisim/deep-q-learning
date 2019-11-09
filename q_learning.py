import numpy as np
import random
from tqdm import tqdm


def q_learning(env,
               episodes,
               max_steps_per_episode,
               alpha=0.1,
               gamma=0.6,
               epsilon=1,
               epsilon_decay=0.1,
               ):
    """
    :param env: Open AI env
    :param episodes: number of episodes
    :param max_steps_per_episode: max steps per episode
    :param alpha: learning rate 𝛼
    :param gamma: discount factor 𝛾,
    :param epsilon: initial epsilon
    :param epsilon_decay: decay rate for decaying epsilon-greedy probability
    :return: Q-table for each state-action pair
    """

    # for ploting
    rewards = []  # todo: reward per episode

    # initialize the Q-table for each state-action pair
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in tqdm(range(1, episodes+1)):
        done = False  # indicating if the returned state is a terminal state
        curr_steps = 0

        state = env.reset()  # get initial state s

        while not done and curr_steps < max_steps_per_episode:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample() # Explore action space
            else:
                action = np.argmax(q_table[state]) # Exploit learned values

            next_state, reward, done, _ = env.step(action)
            if done:
                target = reward
            else:
                target = reward + gamma*np.max(q_table[next_state])

            # update rule for q-value
            q_table[state, action] = (1-alpha)*q_table[state, action] + alpha*target

            # go to next state and increment steps
            state = next_state
            curr_steps += 1

    return q_table


def deep_q_learning():
    pass