#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  19 14:53:25 2019

@author: RaMy
"""

from a2c_agent_phil import Agent as a2c
from environment import ENV
import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch


def main(args):
    # Environment variables
    nb_mec = 3
    nb_vnfs = 2
    # MECs
    min_cpu = 64
    max_cpu = 128
    min_ram = 64
    max_ram = 128
    min_disk = 131072
    max_disk = 524288
    # Containers
    min_c_cpu = 1
    max_c_cpu = 4
    min_c_ram = 1
    max_c_ram = 4
    min_c_disk = 512
    max_c_disk = 4096

    # DQN_agent
    episodes = 10000                        # Total episodes for the training
    batch_size = 32                        # Total used memory in memory replay mode
    max_env_steps = 100                    # Max steps per episode
    batch_update = 4
    # Generate the MEC environment
    env = ENV(nb_mec, nb_vnfs, min_cpu, max_cpu, min_ram, max_ram, min_disk, max_disk, min_c_cpu, max_c_cpu, min_c_ram,
              max_c_ram, min_c_disk, max_c_disk)

    action_size = env.action_space
    print('action_size: {}'.format(action_size))

    state_size = env.observation_space
    print('state_size: {}'.format(state_size))

    if args['observe'] is not None:
        # testing the trained model
        agent = a2c(alpha=0.00001, input_dims=state_size, n_actions=action_size)
        # Loading the Model's weights generated by the selected model
        agent.load_models()
        done = False
        time = 0
        cum_reward = 0
        rewards = []
        state = env.get_state(step=False)
        state = np.reshape(state, [1, state_size])
        while not done:
            env.view_infrastructure_('environment_exploitation.txt', state)
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            env.view_infrastructure_('environment_exploitation.txt', next_state, reward, False)
            next_state = np.reshape(next_state, [state_size])
            state = next_state
            rewards.append(reward)
            cum_reward += reward
            time += 1
        print("sub_episode: {}, reward: {}".format(time, cum_reward))

    else:
        # Training the model
        # Instantiating the DQN_Agent
        agent = a2c(alpha=0.00001, input_dims=state_size, n_actions=action_size)

        tab = {}
        avg_reward_list = []
        rewards = []
        cumulative_rewards = []

        for e in range(episodes):
            state = env.get_state(step=False)
            state = np.reshape(state, [state_size])
            done = False
            time = 0
            cum_reward = 0
            while not done:
                # for time in range(max_env_steps):
                env.view_infrastructure_('environment_exploration.txt', state)
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                # tab[e * time + time] = {"action": action, "reward": reward, "next_state": next_state}
                env.view_infrastructure_('environment_exploration.txt', next_state, reward, False)
                next_state = np.reshape(next_state, [state_size])
                agent.learn(state, reward, next_state, done)
                state = next_state

                rewards.append(reward)
                cum_reward += reward
                time += 1
            # Saving the Model's weights generated by the selected model
            cumulative_rewards.append(cum_reward)
            print("episode: {}/{}, sub_episode: {}, reward: {}".format(e, episodes, time, cum_reward))
            if e > 100:
                avg = np.mean(cumulative_rewards[-100:])
            else:
                avg = np.mean(cumulative_rewards)
            avg_reward_list.append(avg)
            plt.figure(2)
            plt.clf()
            rewards_t = torch.tensor(cumulative_rewards, dtype=torch.float)
            plt.title('Training...')
            plt.xlabel('Episode')
            plt.ylabel('Cumulative reward')
            plt.grid(True)
            plt.plot(rewards_t.numpy())
            # Take 50 episode averages and plot them too
            if len(rewards_t) >= 100:
                means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())
            plt.pause(0.001)  # pause a bit so that plots are updated
            plt.savefig('rewards_live.png')
            plt.close()
            agent.save_models()
        # Plotting the reward/avg_reward
        if args['train'] is not None:
            plt.plot((np.arange(len(avg_reward_list)) + 1), avg_reward_list)
            plt.xlabel('Episodes')
            plt.ylabel('Average Reward')
            plt.title('Average Reward vs Episodes')
            plt.savefig('rewards_{}.png'.format(args['train']))
            plt.close()

            plt.plot(cumulative_rewards)
            plt.plot(avg_reward_list)
            plt.legend(["Reward", "100-episode average"])
            plt.title("Reward history")
            plt.savefig('rewards_{}_2.png'.format(args['train']))
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parsing the type of DRL/RL to be tested')
    parser.add_argument('-t', '--train', help='Train DRL/RL', required=True)
    parser.add_argument('-o', '--observe', help='Observe a trained DRL/RL')
    args = vars(parser.parse_args())
    main(args)