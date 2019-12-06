import torch
import torch.nn.functional as F
from torch.distributions import Normal
from utils import discount_rewards
from models import Policy


class Agent(object):
    def __init__(self, state_space, n_actions, hidden_size=64, gamma=0.98):
        self.train_device = "cpu"
        self.n_actions = n_actions
        self.state_space_dim = state_space
        self.policy_net = Policy(state_space, n_actions, hidden_size, name='a2c_network')
        self.gamma = gamma
        self.states = []
        self.action_probs = []
        self.rewards = []
        self.state_values = []

    def episode_finished(self, episode_number):
        action_probs = torch.stack(self.action_probs, dim=0).to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        state_values = torch.stack(self.state_values, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards, self.state_values = [], [], [], []

        discounted_rewards = discount_rewards(rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        advantages = discounted_rewards - state_values
        loss_critic = F.mse_loss(state_values, discounted_rewards)
        # print(loss_critic)

        weighted_probs = -action_probs * advantages.detach()
        # print(weighted_probs)

        loss = torch.mean(weighted_probs) + loss_critic
        loss.backward()

        if (episode_number+1) % 1 == 0:
            self.policy_net.optimizer.step()
            self.policy_net.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        probabilities, state_value = self.policy_net.forward(observation)
        probabilities = F.softmax(probabilities)
        print(probabilities)
        if evaluation:
            action = torch.argmax(probabilities)
        else:
            action_probs = torch.distributions.Categorical(probabilities)
            action = action_probs.sample()
            self.action_probs.append(action_probs.log_prob(action))
            self.state_values.append(state_value)
        return action.item() + 1

    def store_outcome(self, observation, reward):
        self.states.append(observation)
        self.rewards.append(torch.Tensor([reward]))

    def save_models(self):
        self.policy_net.save_checkpoint()

    def load_models(self):
        self.policy_net.load_checkpoint()

