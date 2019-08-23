#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  13 15:54:23 2019

@author: RaMy
"""
from environment import ENV
import random
import numpy as np


def orchestration():
    # Environment variables
    nb_mec = 3
    nb_vnfs = 3
    # MECs
    min_cpu = 50
    max_cpu = 100
    min_ram = 50
    max_ram = 100
    min_disk = 4096
    max_disk = 12288
    # Containers
    min_c_cpu = 1
    max_c_cpu = 4
    min_c_ram = 1
    max_c_ram = 4
    min_c_disk = 512
    max_c_disk = 4096

    # Agent's variables

    total_episodes = 15000        # Total episodes for the training
    learning_rate = 0.8           # Learning rate
    discount_rate = 0.9           # Discounting rate
    max_steps_episode = 99        # Max steps per episode

    # Exploration parameters
    epsilon = 1.0                 # Exploration rate
    max_epsilon = 1.0             # Exploration probability at start
    min_epsilon = 0.01            # Minimum exploration probability
    decay_rate = 0.005            # Exponential decay rate for exploration prob
    # List of rewards
    rewards = []

    env = ENV(nb_mec, nb_vnfs, min_cpu, max_cpu, min_ram, max_ram, min_disk, max_disk, min_c_cpu, max_c_cpu, min_c_ram,
              max_c_ram, min_c_disk, max_c_disk)
    env.generate_mec()
    env.generate_vnfs()
    env.save_topology("environment")

    action_size = len(env.vnfs) * (len(env.mec) + 2 * 3)
    print('action_size: {}'.format(action_size))
    state_size = (100 ** (3 + 2 * len(env.vnfs))) ** len(env.mec)
    print('state_size: {}'.format(state_size))
    # TODO: too many states to be saved in a matrix, a Deep Q Learning is a must
    qtable = np.zeros((state_size, action_size))
    for episode in range(total_episodes):
        # Reset the environment
        # TODO: reseting always the same topo will allow to converge but exploring the other possibilities is a must
        state = env.get_state(True)  # You should always have the initial generated topology
        step = 0
        total_rewards = 0

        # TODO: for the time being we just gonna consider the static number of state, later it should be modified.
        for step in range(max_steps_episode):
            # Choose an action a in the current environment state (s)
            # First we randomize a number
            exp_exp_tradeoff = random.uniform(0, 1)

            # If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state,:])

            # Else doing a random choice --> exploration
            else:
                action = env.action()

            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])

            total_rewards += reward

            # Our new state is state
            state = new_state

            # If done (if we're dead) : finish episode
            if done == True:
                break

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(total_rewards)


