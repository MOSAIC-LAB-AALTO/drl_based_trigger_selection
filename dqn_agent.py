#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  18 16:51:00 2019

@author: RaMy
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size=None, action_size=None, epsilon_decay=0.995, epsilon=1.0, epsilon_min=0.01,
                 gamma=0.95, alpha=.001, alpha_decay=0.995):

        self.memory = deque(maxlen=1000000)
        self.loss = []
        self.tot_rewards = []
        self.rewards = []
        self.ave_reward_list = []
        self.predict = {}
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon                          # Exploration rate
        self.epsilon_decay = epsilon_decay              # Exponential decay rate for exploration prob
        self.epsilon_min = epsilon_min                  # Minimum exploration probability
        self.gamma = gamma                              # Discounting rate
        self.alpha = alpha                              # Learning rate
        self.alpha_decay = alpha_decay
        self.model = self._build_model()
        self.i = 0
        self.j = 0
        self.action = {}

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(int(np.mean([self.state_size, self.action_size])), activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(1, self.action_size)
        act_values = self.model.predict(state)
        self.action[self.j] = {'full': act_values, 'small': act_values[0], 'max': np.argmax(act_values[0])}
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                q_max = np.amax(self.model.predict(next_state)[0])
                target = (reward + self.gamma * q_max)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.predict[self.i] = {"np.max": q_max, "target": target, "target_f[0][action]": target_f[0][action],
                                    "target_f": target_f}
            self.i += 1
            H = self.model.fit(state, target_f, epochs=1, verbose=0)
            self.loss.append(H.history['loss'])
            self.rewards.append(target)
            self.tot_rewards.append(target)
            if (len(self.rewards)) % 100 == 0:
                ave_reward = np.mean(self.rewards)
                self.ave_reward_list.append(ave_reward)
                self.rewards = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_batch(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in mini_batch:
            target = reward
            if not done:
                # max Q(s',a')
                q_max = np.amax(self.model.predict(next_state)[0])
                target = reward + self.gamma * q_max
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
            self.predict[self.i] = {"np.max": q_max, "target": target, "target_f[0][action]": target_f[0][action],
                                    "target_f": target_f}
            self.i += 1
            self.rewards.append(target)
            if (len(self.rewards)) % batch_size == 0:
                ave_reward = np.mean(self.rewards)
                self.ave_reward_list.append(ave_reward)
                self.rewards = []
        print(states)
        print(targets_f)
        history = self.model.fit(np.array(states), np.array(targets_f), batch_size=len(states), verbose=0)
        self.loss.append(history.history['loss'])
        # Keeping track of loss
        loss = history.history['loss'][0]
        '''print(loss)
        print('epsilon: {}'.format(self.epsilon))
        print('epsilon_decay: {}'.format(self.epsilon_decay))'''

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # The loss is gathered for post usage.

    def test_batch(self, batch_size):
        loss = 0
        Q_sa = 0
        mini_batch = random.sample(self.memory, batch_size)
        state_t, action_t, reward_t, state_t1, terminal = zip(*mini_batch)
        state_t = np.concatenate(state_t)
        state_t1 = np.concatenate(state_t1)
        targets = self.model.predict(state_t)
        Q_sa = self.model.predict(state_t1)
        targets[range(batch_size), action_t] = reward_t + self.gamma*np.max(Q_sa, axis=1)*np.invert(terminal)

        self.ave_reward_list.append(np.mean(reward_t))

        loss += self.model.train_on_batch(state_t, targets)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

