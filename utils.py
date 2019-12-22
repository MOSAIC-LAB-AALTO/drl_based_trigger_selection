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

    def clear(self):
        self.memory_2.append(0)
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def check_end(self):
        """
        Used to determine if the architecture is solved
        :return:
        """
        if len(self.memory_2) == self.win and len(set(self.memory_2)) == 1:
            if min(self.memory_2) == 1:
                self.memory.clear()
                self.memory_2.clear()
                return True
        return False

    def check_win(self):
        """
        Used to limit the number of iteration per episode, this can be omitted but it will results in long training
        and sometimes blocking situations
        :return:
        """
        if len(self.memory) == self.capacity:
            self.memory.clear()
            self.memory_2.append(1)
            return True
        else:
            return False


class WinningCondition(object):
    def __init__(self, capacity=10, limit_rat=70, limit_sct=100):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.limit_rat = limit_rat
        self.limit_sct = limit_sct

    def push(self, args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def new_(self, mec_number, number_resources):
        """
        Used to check if upper limits are broken
        :return:
        """
        rat = []
        sct = []
        for i in range(mec_number * number_resources):
            rat.append(self.memory[0][i])
        for i in range(mec_number * number_resources, len(self.memory[0])):
            sct.append(self.memory[0][i])
        if max(rat) > self.limit_rat or max(sct) > self.limit_sct:
            # print(rat)
            # print(sct)
            return True, -200
        else:
            return False, 0
