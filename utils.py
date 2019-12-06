import random
from collections import namedtuple, deque
import torch

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class EndChecker(object):
    def __init__(self, capacity, limit=30, cap_win=10):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.limit = limit
        self.win = cap_win
        self.memory_2 = deque(maxlen=self.win)

    def push(self, args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def check_end(self):
        if len(self.memory_2) == self.win and len(set(self.memory_2)) == 1:
            if min(self.memory_2[0]) == 1:
                return True
        return False

    def check_win(self):
        if len(self.memory) == self.capacity:
            tmp = []
            for i in range(len(self.memory)):
                tmp.append(max(self.memory[i]))
            if max(tmp) > self.limit:
                print('no')
                self.memory.clear()
                self.memory_2.append(0)
                return False
            else:
                print('yes')
                self.memory.clear()
                self.memory_2.append(1)
                return True
        self.memory_2.append(0)
        return False


class WinningCondition(object):
    def __init__(self, capacity=10, limit=30):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.limit = limit

    def push(self, args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def check_win(self):
        if len(self.memory) == self.capacity:
            tmp = []
            for i in range(len(self.memory)):
                tmp.append(max(self.memory[i]))
            if max(tmp) > self.limit:
                print('no')
                self.memory.clear()
                return False, -50
            else:
                print('yes')
                self.memory.clear()
                return True, 50
        return False, 0

    def new_(self):
        if max(self.memory[0]) > self.limit:
            print('no')
            return True, -200
        else:
            print('yes')
            return False, 0
