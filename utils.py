import numpy as np
import torch

import torch
from torch.utils.data import Dataset

def scale(x, state_low, state_high, action_low, action_high, reward_low, reward_high, a=-1, b=1):
	#(((x[:, 0].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	#(((x[:, 1].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	(((x[:, 2].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	(((x[:, 3].sub_(action_low)).div_((action_high - action_low))).mul_(b-a)).add_(a)
	#(((x[:, 4].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	#(((x[:, 5].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	(((x[:, 6].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	(((x[:, 7].sub_(reward_low)).div_((reward_high - reward_low))).mul_(b-a)).sub_(1.0)
	#(((x[:, 8].sub_(0.0)).div_(1.0)).mul_(b-a)).add_(a)
	return x


def descale(x, state_low, state_high, action_low, action_high, reward_low, reward_high, a=-1, b=1):
	(((x[:, 0].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 1].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 2].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	(((x[:, 3].sub_(a)).div_(b-a)).mul_(action_high - action_low)).add_(action_low)
	(((x[:, 4].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 5].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 6].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	(((x[:, 7].sub_(a)).div_(b-a)).mul_(reward_high - reward_low)).add_(reward_low)
	#((x[:, 8].sub_(a)).div_(b-a)).round_()

	return x


class TrainBatch(Dataset):
	def __init__(self, data, labels):
		self.data = torch.FloatTensor(data)
		self.labels = torch.FloatTensor(labels)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		target = self.labels[index]
		data_val = self.data[index]
		return data_val, target

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
