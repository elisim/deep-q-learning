import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from tqdm import tqdm


def q_learning(env,
               episodes,
               steps_per_episode,
               alpha=0.279458040604177,
               gamma=0.9932380015539644,
               epsilon=1,
               min_epsilon=0.08832403884569218,
               epsilon_decay=0.9999392953141722,
               ):
    """
    :param env: Open AI env
    :param episodes: number of episodes
    :param steps_per_episode: max steps per episode
    :param alpha: learning rate 𝛼
    :param gamma: discount factor 𝛾,
    :param epsilon: initial epsilon
    :param min_epsilon: min epsilon rate (end of decaying)
    :param epsilon_decay: decay rate for decaying epsilon-greedy probability
    :return: Q-table for each state-action pair, and info dict for debugging & plotting
    """
    rewards = []  # todo: reward per episode

    # initialize the Q-table for each state-action pair
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    
    for i in tqdm(range(1, episodes+1)):
        done = False  # indicating if the returned state is a terminal state
        curr_steps = 0  # init current steps for this episode
        state = env.reset()  # get initial state s
        epsilon = max(min_epsilon, epsilon*(epsilon_decay ** i))  # decaying epsilon-greedy probability
        
        while not done and curr_steps < steps_per_episode:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state])  # Exploit learned values
                
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

    info = {"rewards": rewards, "q_table_500_steps": None, "q_table_2000_steps": None}
    return q_table, info
