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
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
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

		self.linear = nn.Linear(self.latent_dim, 2)
		self.fc1 = nn.Linear(self.latent_dim + action_shape, 2)
		self.fc2 = nn.Linear(2, int(np.prod(self.input_shape)))

	def forward(self, z, action):
		activated = F.relu(self.linear(z))
		x = torch.cat((activated, action), dim=1)
		#print(x.shape)
		x = nn.LeakyReLU(0.2, inplace=True)(self.fc1(x))
		x = self.fc2(x)
		x = nn.Tanh()(x)
		x = x.view(x.size(0), *self.input_shape)
		return x

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
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=3):
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

		self.linear = nn.Linear(self.latent_dim, 3)
		self.fc1 = nn.Linear(self.latent_dim + state_shape, 2)
		self.fc2 = nn.Linear(2, int(np.prod(self.input_shape)))

	def forward(self, z, state):
		activated = F.relu(self.linear(z))

		x = torch.cat((activated, state), dim=1)
		x = nn.LeakyReLU(0.2, inplace=True)(self.fc1(x))
		x = self.fc2(x)
		x = nn.Tanh()(x)
		x = x.view(x.size(0), *self.input_shape)
		return x

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


class Discriminator2(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
		super(Discriminator2, self).__init__()

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
			nn.Linear(int(np.prod(self.input_shape)), 3),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(3, 2),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(2, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x_flat = x.view(x.size(0), -1)
		validity = self.model(x_flat)

		return validity


class Discriminator3(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
		super(Discriminator3, self).__init__()

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
			nn.Linear(int(np.prod(self.input_shape)), 1),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x_flat = x.view(x.size(0), -1)
		validity = self.model(x_flat)

		return validity





class JointGAN(object):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=1):
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

	def compute_MMD_loss(self, H_fake, H_real):
		kxx, kxy, kyy = 0, 0, 0
		dividend = 1
		dist_x, dist_y = H_fake / dividend, H_real / dividend
		x_sq = tf.expand_dims(tf.reduce_sum(dist_x ** 2, axis=1), 1)  # 64*1
		y_sq = tf.expand_dims(tf.reduce_sum(dist_y ** 2, axis=1), 1)  # 64*1
		dist_x_T = tf.transpose(dist_x)
		dist_y_T = tf.transpose(dist_y)
		x_sq_T = tf.transpose(x_sq)
		y_sq_T = tf.transpose(y_sq)

		tempxx = -2 * tf.matmul(dist_x, dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2
		tempxy = -2 * tf.matmul(dist_x, dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2
		tempyy = -2 * tf.matmul(dist_y, dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2

		for sigma in [5]:
			kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / (sigma ** 2)))
			kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / (sigma ** 2)))
			kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / (sigma ** 2)))

		# fake_obj = (kxx + kyy - 2*kxy)/n_samples
		# fake_obj = tensor.sqrt(kxx + kyy - 2*kxy)/n_samples
		gan_cost_g = tf.sqrt(kxx + kyy - 2 * kxy)
		return gan_cost_g

	def get_MMD_loss(self, H_fake, H_real):
		return self.compute_MMD_loss(H_fake, H_real)




generator_opt = tf.train.AdamOptimizer(0.0002, beta1=0.5)
discriminator_opt = tf.train.AdamOptimizer(0.0001, beta1=0.5)

adversarial_loss = torch.nn.BCELoss()
Tensor = torch.FloatTensor
softmax_loss = nn.CrossEntropyLoss()
# ----------
#  Training
# ----------



env = gym.make("Pendulum-v0").env

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
n_epochs=30
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_low = env.action_space.low[0]
action_high = env.action_space.high[0]
state_low = env.observation_space.low
state_high = env.observation_space.high
max_action = float(action_high)
batch_size = 512
lr = 0.001
b1 = 0.5
b2 = 0.999
batch_size=256

data=joblib.load('replay2.joblib')
data = scale(torch.FloatTensor(data),state_low, state_high,action_low, action_high)
train_iterator = DataLoader(data, batch_size=batch_size, shuffle=True)

joint = JointGAN(action_dim, state_dim, action_low, action_high, state_low, state_high)
# Optimizers.
optimizer_G1 = torch.optim.Adam(joint.gxy.parameters(), lr=lr, betas=(b1, b2))
optimizer_G2 = torch.optim.Adam(joint.gyx.parameters(), lr=lr, betas=(b1, b2))

optimizer_D1 = torch.optim.Adam(joint.d1.parameters(), lr=lr, betas=(b1, b2))
optimizer_D2 = torch.optim.Adam(joint.d2.parameters(), lr=lr, betas=(b1, b2))
optimizer_D3 = torch.optim.Adam(joint.d3.parameters(), lr=lr, betas=(b1, b2))
optimizer_D4 = torch.optim.Adam(joint.d4.parameters(), lr=lr, betas=(b1, b2))
#optimizer_D_state = torch.optim.Adam(joint.d_state.parameters(), lr=lr, betas=(b1, b2))
#optimizer_D_action = torch.optim.Adam(joint.d_action.parameters(), lr=lr, betas=(b1, b2))

#optimizer_D = torch.optim.Adam(joint.d.parameters(), lr=lr, betas=(b1, b2))


def make_one_hot(labels, C=5):
	y = torch.eye(C)

	target = Variable(y[labels])

	return target

for epoch in range(n_epochs):
	for i, experiences in enumerate(train_iterator):
		# Adversarial ground truths
		states, actions = Tensor(experiences[:, 0:3]).view(-1,state_dim), Tensor(experiences[:,3]).view(-1,action_dim)

		valid_x = Variable(Tensor(experiences.size(0),1).fill_(1.0), requires_grad=False)
		fake_x = Variable(Tensor(experiences.size(0),1).fill_(0.0), requires_grad=False)

		valid_y = Variable(Tensor(experiences.size(0),1).fill_(1.0), requires_grad=False)
		fake_y = Variable(Tensor(experiences.size(0),1).fill_(0.0), requires_grad=False)

		valid_x_valid_y = Variable(Tensor(experiences.size(0), 1).fill_(1.0), requires_grad=False)
		fake_x_fake_y = Variable(Tensor(experiences.size(0), 1).fill_(0.0), requires_grad=False)

		# Configure input
		real_x = Variable(states)
		real_y = Variable(actions)

		# -----------------
		#  Train Generator
		# -----------------

		optimizer_G1.zero_grad()
		optimizer_G2.zero_grad()

		# Sample noise as generator input
		z1 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gxy.latent_dim))))
		z2 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gyx.latent_dim))))
		z3 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gxy.latent_dim))))
		z4 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gyx.latent_dim))))
		z5 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gxy.latent_dim))))
		z6 = Variable(Tensor(np.random.normal(0, 1, (experiences.size(0), joint.gyx.latent_dim))))

		# Generate a batch of images
		#print(descale(experiences[:1],state_low,state_high,action_low,action_high,-20, 0))
		gen_states = joint.gxy(z1, Tensor(experiences.size(0), action_dim).fill_(0))
		#print(descale_state(gen_states[:1],state_low, state_high))
		gen_actions = joint.gyx(z2, Tensor(experiences.size(0), state_dim).fill_(0))
		#print(descale_action(gen_actions[:1], action_low,action_high))

		gen_states_from_actions = joint.gxy(z3, actions)
		#print(descale_state(gen_states_from_actions[:1],state_low,state_high))
		gen_actions_from_states = joint.gyx(z4, states)
		#print(descale_action(gen_actions_from_states[:1], action_low,action_high))

		gen_states_from_gen_actions = joint.gxy(z5, gen_actions)
		# print(descale_state(gen_states_from_actions[:1],state_low,state_high))
		gen_actions_from_gen_states = joint.gyx(z6, gen_states)
		# print(descale_action(gen_actions_from_states[:1], action_low,action_high))

		P1_state, P1_action = states, gen_actions_from_states
		P2_state, P2_action = gen_states_from_actions, actions
		P3_state, P3_action = gen_states, gen_actions_from_gen_states
		P4_state, P4_action = gen_states_from_actions, gen_actions
		P5_state, P5_action = states, actions

		# Loss measures generator's ability to fool the discriminator
		#g_loss1 = adversarial_loss(joint.d_state(gen_states), valid_x)
		#g_loss2 = adversarial_loss(joint.d_action(gen_actions), valid_y)

		g_loss1 = adversarial_loss(joint.d1(P1_state.detach(), P1_action.detach()), valid_x_valid_y)
		g_loss2 = adversarial_loss(joint.d2(P2_state.detach(), P2_action.detach()), valid_x_valid_y)
		g_loss3 = adversarial_loss(joint.d3(P3_state.detach(), P3_action.detach()), valid_x_valid_y)
		g_loss4 = adversarial_loss(joint.d4(P4_state.detach(), P4_action.detach()), valid_x_valid_y)

		g_loss1.backward()
		g_loss2.backward()
		g_loss3.backward()
		g_loss4.backward()

		optimizer_G1.step()
		optimizer_G2.step()

		# ---------------------
		#  Train Discriminator
		# ---------------------

		optimizer_D1.zero_grad()
		optimizer_D2.zero_grad()
		optimizer_D3.zero_grad()
		optimizer_D4.zero_grad()

		# Measure discriminator's ability to classify real from generated samples

		real_loss1 = adversarial_loss(joint.d1(P5_state.detach(), P5_action.detach()), valid_x_valid_y)
		fake_loss1 = adversarial_loss(joint.d1(P1_state.detach(), P1_action.detach()), fake_x_fake_y)
		d_loss1 = (real_loss1 + fake_loss1) / 2

		real_loss2 = adversarial_loss(joint.d2(P5_state.detach(), P5_action.detach()), valid_x_valid_y)
		fake_loss2 = adversarial_loss(joint.d2(P2_state.detach(), P2_action.detach()), fake_x_fake_y)
		d_loss2 = (real_loss2 + fake_loss2) / 2

		real_loss3 = adversarial_loss(joint.d3(P5_state.detach(), P5_action.detach()), valid_x_valid_y)
		fake_loss3 = adversarial_loss(joint.d3(P3_state.detach(), P3_action.detach()), fake_x_fake_y)
		d_loss3 = (real_loss3 + fake_loss3) / 2

		real_loss4 = adversarial_loss(joint.d4(P5_state.detach(), P5_action.detach()), valid_x_valid_y)
		fake_loss4 = adversarial_loss(joint.d4(P4_state.detach(), P4_action.detach()), fake_x_fake_y)
		d_loss4 = (real_loss4 + fake_loss4) / 2

		d_loss1.backward()
		d_loss2.backward()
		d_loss3.backward()
		d_loss4.backward()
		optimizer_D1.step()
		optimizer_D2.step()
		optimizer_D3.step()
		optimizer_D4.step()


		d_loss = d_loss1 + d_loss2 + d_loss3 + d_loss4
		g_loss = g_loss1 + g_loss2 + g_loss3 + g_loss4


		print(
			"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
			% (epoch, n_epochs, i, len(train_iterator), d_loss.item(), g_loss.item())
		)

		batches_done = epoch * len(train_iterator) + i
