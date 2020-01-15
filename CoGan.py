import torch
import torch.nn
import torch.optim as optim
import numpy as np
import joblib
import gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.autograd import Variable
from tensorflow.examples.tutorials.mnist import input_data
import scipy.ndimage.interpolation


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)
mb_size = 32
z_dim = 5
X_dim = 3
y_dim = 1
h_dim = 5
cnt = 0
lr = 1e-2


""" Shared Generator weights """
G_shared = torch.nn.Sequential(
	torch.nn.Linear(z_dim, h_dim),
	torch.nn.ReLU(),
)

""" Generator 1 """
G1_ = torch.nn.Sequential(
	torch.nn.Linear(h_dim, X_dim),
	torch.nn.Sigmoid()
)

""" Generator 2 """
G2_ = torch.nn.Sequential(
	torch.nn.Linear(h_dim, X_dim),
	torch.nn.Sigmoid()
)


def G1(z):
	h = G_shared(z)
	X = G1_(h)
	return X


def G2(z):
	h = G_shared(z)
	X = G2_(h)
	return X


""" Shared Discriminator weights """
D_shared = torch.nn.Sequential(
	torch.nn.Linear(h_dim, 1),
	torch.nn.Sigmoid()
)

""" Discriminator 1 """
D1_ = torch.nn.Sequential(
	torch.nn.Linear(X_dim, h_dim),
	torch.nn.ReLU()
)

""" Discriminator 2 """
D2_ = torch.nn.Sequential(
	torch.nn.Linear(X_dim, h_dim),
	torch.nn.ReLU()
)


def D1(X):
	h = D1_(X)
	y = D_shared(h)
	return y


def D2(X):
	h = D2_(X)
	y = D_shared(h)
	return y


D_params = (list(D1_.parameters()) + list(D2_.parameters()) +
			list(D_shared.parameters()))
G_params = (list(G1_.parameters()) + list(G2_.parameters()) +
			list(G_shared.parameters()))
nets = [G_shared, G1_, G2_, D_shared, D1_, D2_]


def reset_grad():
	for net in nets:
		net.zero_grad()


G_solver = optim.Adam(G_params, lr=lr)
D_solver = optim.Adam(D_params, lr=lr)

X_train = mnist.train.images
half = int(X_train.shape[0] / 2)

# Real image
X_train1 = X_train[:half]
# Rotated image
X_train2 = X_train[half:].reshape(-1, 28, 28)
X_train2 = scipy.ndimage.interpolation.rotate(X_train2, 90, axes=(1, 2))
X_train2 = X_train2.reshape(-1, 28*28)

# Cleanup
del X_train

def scale(x, state_low, state_high, action_low, action_high, reward_low, reward_high, a=-1, b=1):
	(((x[:, 0].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	(((x[:, 1].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	(((x[:, 2].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	(((x[:, 3].sub_(action_low)).div_((action_high - action_low))).mul_(b-a)).add_(a)
	(((x[:, 4].sub_(state_low[0])).div_((state_high[0] - state_low[0]))).mul_(b-a)).add_(a)
	(((x[:, 5].sub_(state_low[1])).div_((state_high[1] - state_low[1]))).mul_(b-a)).add_(a)
	(((x[:, 6].sub_(state_low[2])).div_((state_high[2] - state_low[2]))).mul_(b-a)).add_(a)
	(((x[:, 7].sub_(reward_low)).div_((reward_high - reward_low))).mul_(b-a)).sub_(1.0)
	(((x[:, 8].sub_(0.0)).div_(1.0)).mul_(b-a)).add_(a)
	return x

def sample_x(X, size):
	start_idx = np.random.randint(0, X.shape[0]-size)
	return Variable(torch.from_numpy(X[start_idx:start_idx+size]))



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
lr = 0.0002
b1 = 0.5
b2 = 0.999
batch_size=256
data=joblib.load('replay2.joblib')
data = scale(torch.FloatTensor(data),state_low, state_high,action_low, action_high)

for it in range(100000):
	X1 = sample_x(X_train1, mb_size)
	X2 = sample_x(X_train2, mb_size)
	z = Variable(torch.randn(mb_size, z_dim))

	# Dicriminator
	G1_sample = G1(z)
	D1_real = D1(X1)
	D1_fake = D1(G1_sample)

	G2_sample = G2(z)
	D2_real = D2(X2)
	D2_fake = D2(G2_sample)

	D1_loss = torch.mean(-torch.log(D1_real + 1e-8) -
						 torch.log(1. - D1_fake + 1e-8))
	D2_loss = torch.mean(-torch.log(D2_real + 1e-8) -
						 torch.log(1. - D2_fake + 1e-8))
	D_loss = D1_loss + D2_loss

	D_loss.backward()

	# Average the gradients
	for p in D_shared.parameters():
		p.grad.data = 0.5 * p.grad.data

	D_solver.step()
	reset_grad()

	# Generator
	G1_sample = G1(z)
	D1_fake = D1(G1_sample)

	G2_sample = G2(z)
	D2_fake = D2(G2_sample)

	G1_loss = torch.mean(-torch.log(D1_fake + 1e-8))
	G2_loss = torch.mean(-torch.log(D2_fake + 1e-8))
	G_loss = G1_loss + G2_loss

	G_loss.backward()

	# Average the gradients
	for p in G_shared.parameters():
		p.grad.data = 0.5 * p.grad.data

	G_solver.step()
	reset_grad()

	# Print and plot every now and then
	if it % 1000 == 0:
		print('Iter-{}; D1_loss: {:.4}; G1_loss: {:.4}; '
			  'D2_loss: {:.4}; G2_loss: {:.4}'
			  .format(
				  it, D1_loss.data[0], G1_loss.data[0],
				  D2_loss.data[0], G2_loss.data[0])
			  )

		z = Variable(torch.randn(8, z_dim))
		samples1 = G1(z).data.numpy()
		samples2 = G2(z).data.numpy()
		samples = np.vstack([samples1, samples2])

		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

		if not os.path.exists('out/'):
			os.makedirs('out/')

		plt.savefig('out/{}.png'
					.format(str(cnt).zfill(3)), bbox_inches='tight')
		cnt += 1
		plt.close(fig)