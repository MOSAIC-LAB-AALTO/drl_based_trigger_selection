#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  19 14:57:00 2019

@author: RaMy
"""

import random
import numpy as np
from utils import Transition, ReplayMemory
from models import GenericNetwork
import torch as T
import torch.nn.functional as F


class Agent:
    def __init__(self, state_space, n_actions, replay_buffer_size=50000,
                 batch_size=32, hidden_size=64, gamma=0.99):
        self.n_actions = n_actions
        self.state_space_dim = state_space
        self.policy_net = GenericNetwork(state_space, n_actions, hidden_size, name='dqn_network_')
        self.target_net = GenericNetwork(state_space, n_actions, hidden_size, name='target_dqn_network_')
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.action = {}
        self.j = 0

    def learn(self):
        """
        Learning function
        :return:
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        non_final_mask = 1-T.tensor(batch.done, dtype=T.uint8)

        # avoid having an empty tensor
        test_tensor = T.zeros(self.batch_size)
        while T.all(T.eq(test_tensor, non_final_mask)).item() is True:
            transitions = self.memory.sample(self.batch_size)
            batch = Transition(*zip(*transitions))
            non_final_mask = 1-T.tensor(batch.done, dtype=T.uint8)

        non_final_next_states = [s for nonfinal,s in zip(non_final_mask, batch.next_state) if nonfinal > 0]
        non_final_next_states = T.stack(non_final_next_states)
        state_batch = T.stack(batch.state)
        action_batch = T.cat(batch.action)
        reward_batch = T.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = T.zeros(self.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute mse loss
        loss = F.mse_loss(state_action_values.squeeze(), expected_state_action_values)
        # Optimize the model
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.policy_net.optimizer.step()

    def get_action(self, state, epsilon=0.05):
        """
        Used to select actions
        :param state:
        :param epsilon:
        :return: action
        """
        sample = random.random()
        if sample > epsilon:
            with T.no_grad():
                state = T.from_numpy(state).float()
                q_values = self.policy_net(state)
                self.action[self.j] = {'list_of_actions': q_values, 'max': T.argmax(q_values).item()}
                self.j += 1
                return T.argmax(q_values).item() + 1
        else:
            action = random.randrange(self.n_actions)
            return action + 1

    def update_target_network(self):
        """
        Used to update target networks
        :return:
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, state, action, reward, next_state, done):
        """
        Used for memory replay purposes
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        action = T.Tensor([[action]]).long()
        reward = T.tensor([reward], dtype=T.float32)
        next_state = T.from_numpy(next_state).float()
        state = T.from_numpy(state).float()
        self.memory.push(state, action, reward, next_state, done)

    def save_models(self):
        """
        Used to save models
        :return:
        """
        self.policy_net.save_checkpoint()
        self.target_net.save_checkpoint()

    def load_models(self):
        """
        Used to load models
        :return:
        """
        self.policy_net.load_checkpoint()
        # self.target_net.load_checkpoint()
