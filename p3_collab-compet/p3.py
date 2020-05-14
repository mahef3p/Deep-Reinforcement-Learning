from unityagents import UnityEnvironment
import time
import sys
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

def train_ddpg(dev, weights_file_actor, weights_file_critic, n_episodes=1000, max_t=1000):
    """DDPG Learning.

    Params
    ======
        dev (string): cpu or gpu
        weights_file_actor (string): name of the file to save the weights of the actor
        weights_file_critic (string): name of the file to save the weights of the critic
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores = []    # list containing scores from each episode (average of all the agents)
    averages = []  # list containing averages of the scores. Position i (1-index) has the average of the last min(i,100) episodes
    scores_window = deque(maxlen=100)  # last 100 averaged scores for all the agents
    env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    agent = Agent(state_size, action_size, random_seed=0, device=dev)

    print('Number of agents: {:d}'.format(num_agents))
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()  # reset noise for the actions
        states = env_info.vector_observations
        current_scores = np.zeros(num_agents)  # initialize the score for all the agents
        for t in range(max_t):
            actions = agent.act(states)  # process the states of all the agents

            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            current_scores += rewards
            if np.any(dones):
                break
        max_score = np.max(current_scores)  # current maximum score of all the agents
        scores.append(max_score)
        scores_window.append(max_score)
        averages.append(np.mean(scores_window))
        if (i_episode % 100 != 0):
            print('\rEpisode {}\tScore: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, max_score, averages[i_episode-1]), end="")
        else:
            print('\rEpisode {}\tScore: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, max_score, averages[i_episode-1]))
        if (averages[i_episode-1] >= 0.5):
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode-100, averages[i_episode-1]))
            torch.save(agent.actor_local.state_dict(), weights_file_actor)
            torch.save(agent.critic_local.state_dict(), weights_file_critic)
            break
 
    env.close()
    return scores, averages


def test(dev, weights_file_actor, weights_file_critic, n_episodes=100, max_t=1000):
    """Test the environment with the parameters stored in checkpoint.pth

    Params
    ======
        dev (string): cpu or gpu
        weights_file_actor (string): name of the file to load the weights of the actor
        weights_file_critic (string): name of the file to load the weights of the critic
        n_episodes (int): number of test episodes that will be performed
    """
    env = UnityEnvironment(file_name='./Tennis_Linux/Tennis.x86_64')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]
    action_size = brain.vector_action_space_size
    agent = Agent(state_size, action_size, random_seed=0, device=dev)
    scores = []

    # load the weights from file
    print('Number of agents: {:d}'.format(num_agents))
    print('Loading weights')
    try:
        checkpoint_actor = torch.load(weights_file_actor)
    except FileNotFoundError:
        print('Error: File \'{}\' not found'.format(weights_file_actor))
        sys.exit(1)
    try:
        checkpoint_critic = torch.load(weights_file_critic)
    except FileNotFoundError:
        print('Error: File \'{}\' not found'.format(weights_file_critic))
        sys.exit(1)

    agent.actor_local.load_state_dict(checkpoint_actor)
    agent.critic_local.load_state_dict(checkpoint_critic)
    print('Running {} episodes'.format(n_episodes))
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name]
        current_scores = np.zeros(num_agents)  # initialize the score for all the agents
        states = env_info.vector_observations
        for t in range(max_t):
            actions = agent.act(states, add_noise=False)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            current_scores += rewards
            if np.any(dones):
                break 
        max_score = np.max(current_scores)  # current maximum score of all the agents
        scores.append(max_score)
        if (i_episode % 100 != 0):
            print('\rEpisode {}\tScore: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, max_score, np.mean(scores)), end="")
        else:
            print('\rEpisode {}\tScore: {:.3f}\tAverage Score: {:.3f}'.format(i_episode, max_score, np.mean(scores)))

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

    weights_file_actor = 'model_actor.pth'
    weights_file_critic = 'model_critic.pth'
    perform_train = True if (sys.argv[1] == 'train') else False
    if (perform_train):
        t_start = time.time()
        scores, averages = train_ddpg(device, weights_file_actor, weights_file_critic, n_episodes=1000, max_t=1000)
        t_total = time.time() - t_start
        print('\nTime: {:.2f} sec.'.format(t_total))

        # plot the scores and the averages
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores, color='dodgerblue', label='Score')
        plt.plot(np.arange(len(scores)-49), averages[49:], color='r', label='Average')
        plt.legend(loc='best')
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()
    else:
        test(device, weights_file_actor, weights_file_critic, n_episodes=100)
    sys.exit(0)

if __name__ == '__main__':
    main()

