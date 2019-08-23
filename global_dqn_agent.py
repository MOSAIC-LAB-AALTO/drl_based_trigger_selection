#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  21 12:31:00 2019

@author: RaMy
"""

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.initializers import glorot_normal, normal
import json


class GlobalDQNAgent:
    def __init__(self, state_size=None, action_size=None, epsilon_decay=0.99, epsilon=1.0, epsilon_min=0.01,
                 gamma=0.95, alpha=.05, alpha_decay=0.995):

        self.memory = deque(maxlen=100000)
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = epsilon                          # Exploration rate
        self.epsilon_decay = epsilon_decay              # Exponential decay rate for exploration prob
        self.epsilon_min = epsilon_min                  # Minimum exploration probability
        self.gamma = gamma                              # Discounting rate
        self.alpha = alpha                              # Learning rate
        self.alpha_decay = alpha_decay
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.loss = []
        self.tot_rewards = []
        self.rewards = []
        self.ave_reward_list = []
        self.predict = {}
        self.i = 0
        self.j = 0
        self.action = {}
        self.glorot_initializer = glorot_normal(seed=42)
        self.normal_initializer = normal(seed=42)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu', kernel_initializer='glorot_normal'))
        model.add(Dense(int(np.mean([self.state_size, self.action_size])), activation='relu', kernel_initializer='normal'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        first_layer_weights = model.layers[0].get_weights()[0]
        print('THE WEIGHTS OF THE FIRST LAYER ARE: {}'.format(first_layer_weights))
        model.summary()
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(1, self.action_size)
        act_values = self.model.predict(state)
        self.action[self.j] = {'full': act_values, 'small': act_values[0], 'max': np.argmax(act_values[0])}
        return np.argmax(act_values[0])  # returns action

    def replay_dqn(self, batch_size):
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

    def replay_dqn_batch(self, batch_size):
        pass


    def replay_fixed_target_dqn(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                q_max = np.amax(self.target_model.predict(next_state)[0])
                target[0][action] = (reward + self.gamma * q_max)
            self.predict[self.i] = {"np.max": q_max, "target_f[0][action]": target[0][action], "target": target}
            self.i += 1
            H = self.model.fit(state, target, epochs=1, verbose=0)
            self.loss.append(H.history['loss'])
            self.rewards.append(target)
            self.tot_rewards.append(target)
            if (len(self.rewards)) % 100 == 0:
                ave_reward = np.mean(self.rewards)
                self.ave_reward_list.append(ave_reward)
                self.rewards = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_fixed_target_dqn_batch(self, batch_size):
        pass

    def replay_double_dqn(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in mini_batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                predicted_value = self.model.predict(next_state)
                best_action = np.argmax(predicted_value)
                # print('the predicted value is: {}'.format(predicted_value))
                # print('the best action is: {}'.format(best_action))
                double_q_value = self.target_model.predict(next_state)[0]
                # print('the predicted values: {}'.format(double_q_value))
                # print('the value of the action {} is: {}'.format(best_action, double_q_value[best_action]))
                target[0][best_action] = (reward + self.gamma * double_q_value[best_action])

            self.predict[self.i] = {
                "target[0][action]": target[0][action],
                "target_best_action": double_q_value[best_action],
                "target[0][best_action]": target[0][best_action]
            }
            self.i += 1
            H = self.model.fit(state, target, epochs=1, verbose=0)
            self.loss.append(H.history['loss'])
            self.rewards.append(target[0][best_action])
            self.tot_rewards.append(target)
            if (len(self.rewards)) % 100 == 0:
                ave_reward = np.mean(self.rewards)
                self.ave_reward_list.append(ave_reward)
                self.rewards = []
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay_double_dqn_batch(self, batch_size):
        pass


    def load(self, name):
        self.model.load_weights("model_{}.h5".format(name))
        self.model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        print("Outputting all the Loaded weights")
        print(self.model.get_weights())


    def save(self, name):
        self.model.save_weights(name + ".h5", overwrite=True)
        with open(name + ".json", "w") as outfile:
            json.dump(self.model.to_json(), outfile)
        print("Outputting all the Saved weights")
        print(self.model.get_weights())


