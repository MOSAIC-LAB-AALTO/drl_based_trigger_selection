import torch as T
import torch.nn.functional as F
from models import ActorCriticNetwork


class Agent(object):
    def __init__(self, alpha, input_dims, gamma=0.99,
                 layer1_size=64, layer2_size=256, n_actions=2):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                               layer2_size, n_actions=n_actions, name='a2c_network_')

        self.log_probs = None

    def get_action(self, observation):
        """
        Used to select actions
        :param state:
        :param epsilon:
        :return:
        """
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities, dim=-1)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        log_probs = action_probs.log_prob(action)
        self.log_probs = log_probs

        return action.item() + 1

    def learn(self, state, reward, new_state, done):
        """
        Learning function
        :return:
        """
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)
        reward = T.tensor(reward, dtype=T.float).to(self.actor_critic.device)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2

        (actor_loss + critic_loss).backward()

        self.actor_critic.optimizer.step()

    def save_models(self):
        """
        Used to save models
        :return:
        """
        self.actor_critic.save_checkpoint()

    def load_models(self):
        """
        Used to load models
        :return:
        """
        self.actor_critic.load_checkpoint()
