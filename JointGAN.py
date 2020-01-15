import numpy as np
import torch.nn as nn
import torch
import tensorflow as tf
from torch.autograd import Variable
import torch.nn.functional as F
import gym
from torch.utils.data import DataLoader

import joblib

def scale(x, state_low, state_high, action_low, action_high, a=-1, b=1):
	(((x[:, 0].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	(((x[:, 1].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	(((x[:, 2].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	(((x[:, 3].sub_(action_low)).div_((action_high - action_low))).mul_(b-a)).add_(a)
	#(((x[:, 4].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	#(((x[:, 5].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	#(((x[:, 6].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	#(((x[:, 7].sub_(reward_low)).div_((reward_high - reward_low))).mul_(b-a)).sub_(1.0)
	#(((x[:, 8].sub_(0.0)).div_(1.0)).mul_(b-a)).add_(a)
	return x


def descale(y, state_low, state_high, action_low, action_high, reward_low, reward_high, a=-1, b=1):
	x=y.clone()
	(((x[:, 0].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 1].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 2].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	(((x[:, 3].sub_(a)).div_(b-a)).mul_(action_high - action_low)).add_(action_low)
	#(((x[:, 4].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	#(((x[:, 5].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	#(((x[:, 6].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	#(((x[:, 7].sub_(a)).div_(b-a)).mul_(reward_high - reward_low)).add_(reward_low)
	#((x[:, 8].sub_(a)).div_(b-a)).round_()


	return x


def descale_state(y, state_low, state_high, a=-1, b=1):
	x = y.clone()
	(((x[:, 0].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 1].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 2].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	return x

def descale_action(y, action_low, action_high, a=-1, b=1):
	x = y.clone()
	(((x.sub_(a)).div_(b-a)).mul_(action_high - action_low)).add_(action_low)
	return x


class Generator1(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=5):
		super(Generator1, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.state_shape
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.latent_dim= latent_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_shape = (self.feature_size,)

		self.model = nn.Sequential(
			nn.Linear(self.action_shape + self.latent_dim, 5),
			nn.LeakyReLU(0.2),
			nn.Linear(5, self.state_shape),
			nn.Tanh()
		)

	def forward(self, z, action):
		x = torch.cat((z, action), dim=1)
		img = self.model(x)
		img = img.view(-1, self.state_shape)
		return img

	def sample(self, batch_size, actions):
		# Sample noise as generator input
		z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))

		# Generate a batch of images
		gen_experiences = self.model(z, actions).detach().numpy()

		a=-1
		b=1
		(((gen_experiences[:, 0].sub_(a)).div_(b - a)).mul_(self.state_high[0] - self.state_low[0])).add_(self.state_low[0])
		(((gen_experiences[:, 1].sub_(a)).div_(b - a)).mul_(self.state_high[1] - self.state_low[1])).add_(self.state_low[1])
		(((gen_experiences[:, 2].sub_(a)).div_(b - a)).mul_(self.state_high[2] - self.state_low[2])).add_(self.state_low[2])
		result = gen_experiences

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
		)


class Generator2(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=5):
		super(Generator2, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.latent_dim= latent_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_shape = (self.feature_size,)

		self.model = nn.Sequential(
			nn.Linear(self.state_shape + self.latent_dim, 5),
			nn.LeakyReLU(0.2),
			nn.Linear(5, self.action_shape),
			nn.Tanh()
		)

	def forward(self, z, state):
		x = torch.cat((z, state), dim=1)
		img = self.model(x)
		img = img.view(-1, self.action_shape)
		return img

	def sample(self, batch_size, states):
		# Sample noise as generator input
		z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))

		# Generate a batch of images
		a=-1
		b=1
		gen_experiences = self.model(z, states).detach().numpy()
		(((gen_experiences.sub_(a)).div_(b - a)).mul_(self.action_high - self.action_low)).add_(self.action_low)
		result = gen_experiences

		return (
			torch.FloatTensor(result).to(self.device)
		)


class Discriminator(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
		super(Discriminator, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + self.state_shape
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.latent_dim= latent_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_shape = (self.feature_size,)

		self.model = nn.Sequential(
			nn.Linear(int(np.prod(self.input_shape)), 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
			nn.Sigmoid(),
		)

	def forward(self, x, y):
		x = torch.cat((x, y), 1)
		x_flat = x.view(x.size(0), -1)
		validity = self.model(x_flat)

		return validity


class DiscriminatorX(nn.Module):

	def __init__(self, state_shape):
		super(DiscriminatorX, self).__init__()
		self.state_shape = state_shape
		self.model = nn.Sequential(
			nn.Linear(self.state_shape, 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
		)

	def forward(self, x1):
		# Concatenate label embedding and image to produce input
		validity = self.model(x1)
		return validity


class DiscriminatorY(nn.Module):
	def __init__(self, action_shape):
		super(DiscriminatorY, self).__init__()
		self.action_shape = action_shape
		self.model = nn.Sequential(
			nn.Linear(self.action_shape, 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
		)

	def forward(self, y1):
		# Concatenate label embedding and image to produce input
		validity = self.model(y1)
		return validity




class JointGAN(object):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=5):
		super(JointGAN, self).__init__()
		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + (2 * self.state_shape) + 1
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.latent_dim = latent_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_shape = (self.feature_size,)

		self.gxy = Generator1(action_shape, state_shape, action_low, action_high, state_low, state_high)
		self.gyx = Generator2(action_shape, state_shape, action_low, action_high, state_low, state_high)

		self.d1 = Discriminator(action_shape, state_shape, action_low, action_high, state_low,state_high)
		self.d2 = Discriminator(action_shape, state_shape, action_low, action_high, state_low, state_high)
		self.d3 = Discriminator(action_shape, state_shape, action_low, action_high, state_low, state_high)
		self.d4 = Discriminator(action_shape, state_shape, action_low, action_high, state_low, state_high)
		#self.d_state = Discriminator2(action_shape, state_shape, action_low, action_high, state_low, state_high)
		#self.d_action = Discriminator3(action_shape, state_shape, action_low, action_high, state_low, state_high)

	def descale(self, x):
		(((x[:, 0].add_(1.0)).div_(2.0)).mul_(self.state_high[0] - self.state_low[0])).add_(self.state_low[0])
		(((x[:, 1].add_(1.0)).div_(2.0)).mul_(self.state_high[1] - self.state_low[1])).add_(self.state_low[1])
		(((x[:, 2].add_(1.0)).div_(2.0)).mul_(self.state_high[2] - self.state_low[2])).add_(self.state_low[2])
		(((x[:, 3].add_(1.0)).div_(2.0)).mul_(self.action_high - self.action_low)).add_(self.action_low)

		return x

	def sample(self, size):
		noise = torch.randn(size, self.latent_dim)
		z = torch.zeros((size, self.action_shape))
		state = self.gxy(noise,z).detach()
		action = self.gyx(noise,state).detach()

		result = self.descale(torch.cat((state, action), 1))

		return result

	def get_next(self, state, action):
		# print(state.shape)
		th = state[0]
		thdot = state[2]

		g = 10.0
		m = 1.
		l = 1.
		dt = 0.05

		action = np.clip(action, -2.0, 2.0).item()
		costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (action ** 2)

		newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l ** 2) * action) * dt
		newth = th + newthdot * dt
		newthdot = np.clip(newthdot, -8, 8)  # pylint: disable=E1111
		new_state = [np.cos(newth), np.sin(newth), newthdot]

		return new_state, [-costs], [0]

	def sampleReplay(self, batch_size):
		samples = self.sample(batch_size)  # generate from genReplay somehow
		state = samples[:, 0:3]
		action = samples[:, -1]
		next_state = []
		reward = []
		not_done = []
		for ix in range(batch_size):
			ns, r, nd = self.get_next(state[ix], action[ix])
			next_state.append(ns)
			reward.append(r)
			not_done.append(nd)
		next_state = torch.Tensor(next_state)
		reward = torch.Tensor(reward)
		not_done = torch.Tensor(not_done)
		action = action.unsqueeze(1)

		return (
			torch.FloatTensor(state),
			torch.FloatTensor(action),
			torch.FloatTensor(next_state),
			torch.FloatTensor(reward),
			torch.FloatTensor(not_done)
		)

def angle_normalize(x):
	return (((x + np.pi) % (2 * np.pi)) - np.pi)
