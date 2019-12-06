#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  19 10:46:30 2019

@author: RaMy
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class GenericNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer=400, learning_rate=5e-4, name='generic', chkpt_dir=''):
        super(GenericNetwork, self).__init__()
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.input_dims = state_dim
        self.hidden_layer = hidden_layer
        self.action_dim = action_dim
        self.fc1 = nn.Linear(self.input_dims, self.hidden_layer)
        self.fc2 = nn.Linear(self.hidden_layer, self.hidden_layer)
        self.fc3 = nn.Linear(self.hidden_layer, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        # x = torch.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space, hidden_layer=256, learning_rate=1e-5, name='generic', chkpt_dir=''):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = hidden_layer
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.fc1 = torch.nn.Linear(state_space, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)
        self.fc_pi = torch.nn.Linear(self.hidden, action_space)
        self.fc_value = torch.nn.Linear(self.hidden, 1)
        # self.init_weights()

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, state):
        state = torch.Tensor(state).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.fc_pi(x)
        v = self.fc_value(x)
        return pi, v

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))


class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name='generic', chkpt_dir=''):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name)
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.pi = nn.Linear(self.fc2_dims, n_actions)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = 'cpu'  # T.device('cuda:0' if T.cuda.is_available() else 'cuda:1')
        self.to(self.device)

    def forward(self, observation):
        state = torch.Tensor(observation).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        pi = self.pi(x)
        v = self.v(x)
        return pi, v

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
