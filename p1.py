from unityagents import UnityEnvironment
import sys
import time
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent

def train_dqn(dev, weights_file, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        dev (string): cpu or gpu
        weights_file (string): name of the file to save the weights
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []    # list containing scores from each episode
    averages = []  # list containing averages of the scores. Position i (1-index) has the average of the last min(i,100) episodes
    scores_window = deque(maxlen=100)  # last 100 scores
    env = UnityEnvironment(file_name='./Banana_Linux/Banana.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    agent = Agent(state_size, action_size, seed=0, device=dev)

    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)        # save most recent score
        scores.append(score)               # save most recent score
        averages.append(np.mean(scores_window))
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        if (i_episode % 100 != 0):
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, averages[i_episode-1]), end="")
        else:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, averages[i_episode-1]))
        if (averages[i_episode-1] >= 13.0):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, averages[i_episode-1]))
            torch.save(agent.qnetwork_local.state_dict(), weights_file)
            break
 
    env.close()
    return scores, averages


def test(dev, weights_file, n_episodes=100, max_t=1000):
    """Test the environment with the parameters stored in weights_file

    Params
    ======
        dev (string): cpu or gpu
        weights_file (string): name of the file to load the weights
        n_episodes (int): number of test episodes that will be performed
        max_t (int): maximum number of timesteps per episode
    """
    env = UnityEnvironment(file_name='./Banana_Linux/Banana.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    agent = Agent(state_size, action_size, seed=0, device=dev)

    # load the weights from file
    print('Loading weights')
    try:
        checkpoint = torch.load(weights_file)
    except FileNotFoundError:
        print('Error: File \'{}\' not found'.format(weights_file))
        sys.exit(1)

    agent.qnetwork_local.load_state_dict(checkpoint)
    scores = []
    print('Running {} episodes'.format(n_episodes))
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        score = 0
        state = env_info.vector_observations[0]
        for j in range(max_t):
            action = agent.act(state)
            env_info = env.step(action)[brain_name]
            state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            score += reward
            if done:
                break 
        scores.append(score)
        if (i_episode % 100 != 0):
            print('\rEpisode {}\tScore: {:.0f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores)), end="")
        else:
            print('\rEpisode {}\tScore: {:.0f}\tAverage Score: {:.2f}'.format(i_episode, score, np.mean(scores)))

    env.close()


def main():
    device = "cpu"
    nargs = len(sys.argv)
    if (nargs < 2):
        print('Usage: {} train|test [cpu|gpu]'.format(str(sys.argv[0])))
        sys.exit(1)

    if ((sys.argv[1] != 'train') and (sys.argv[1] != 'test')):
        print('Usage: {} train|test [cpu|gpu]'.format(str(sys.argv[0])))
        sys.exit(1)

    if (nargs >= 3):
        if ((sys.argv[2] != 'cpu') and (sys.argv[2] != 'gpu')):
            print('Usage: {} train|test [cpu|gpu]'.format(str(sys.argv[0])))
            sys.exit(1)
        if (sys.argv[2] == 'gpu'):
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
            else:
                print('Warning: CUDA not available. Running on cpu')

    weights_file = 'model.pth'
    perform_train = True if (sys.argv[1] == 'train') else False
    if (perform_train):
        t_start = time.time()
        scores, averages = train_dqn(device, weights_file, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995)
        t_total = time.time() - t_start
        print('\nTime: {:.2f} sec.'.format(t_total))

        # plot the scores and the averages
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores, color='dodgerblue', label='Score')
        plt.plot(np.arange(len(scores)-99), averages[99:], color='r', label='Average')
        plt.legend(loc='best')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        test(device, weights_file, n_episodes=100, max_t=1000)
    sys.exit(0)

if __name__ == '__main__':
    main()

