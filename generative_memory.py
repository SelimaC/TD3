import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
from utils import scale, descale


class VAE(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, h_dim=7, z_dim=3):
		super(VAE, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + (2 * self.state_shape) + 2
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.z_dim = z_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


		self.encoder = nn.Sequential(
			nn.Linear(self.feature_size, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, z_dim * 2)
		)

		self.decoder = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, self.feature_size),
			nn.Tanh()
		)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = torch.randn_like(std)
		z = mu + std * eps
		return z

	def forward(self, x):

		h = self.encoder(x)
		mu, logvar = torch.chunk(h, 2, dim=1)
		z = self.reparameterize(mu, logvar)
		z = self.decoder(z)

		return z, mu, logvar

	def sample(self, batch_size):

		sample = Variable(torch.randn(batch_size, self.z_dim))
		recon_x = self.decoder(sample).detach().numpy()
		result = descale(torch.Tensor(recon_x), self.state_low, self.state_high,
		                 self.action_low, self.action_high, self.reward_low, self.reward_high)

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, -1]).unsqueeze(1).to(self.device)
		)

	def sampleMemory(self, batch_size):

		sample = Variable(torch.randn(batch_size, self.z_dim))
		recon_x = self.decoder(sample).detach().numpy()
		result = self.descale(torch.Tensor(recon_x))

		## descale

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
			torch.FloatTensor(np.random.choice(2,batch_size, p=[1/200.0, 199/200.0])).unsqueeze(1).to(self.device)
		)


class Generator(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
		super(Generator, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + (2 * self.state_shape) + 1
		self.action_low = action_low
		self.action_high = action_high
		self.state_low = state_low
		self.state_high = state_high
		self.reward_low = -20.0
		self.reward_high = 0.0
		self.latent_dim= latent_dim
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.input_shape = (self.feature_size,)

		def block(in_feat, out_feat, normalize=True):
			layers = [nn.Linear(in_feat, out_feat)]
			if normalize:
				layers.append(nn.BatchNorm1d(out_feat, 0.8))
			layers.append(nn.LeakyReLU(0.2, inplace=True))
			return layers

		self.model = nn.Sequential(
			*block(self.latent_dim, 4, normalize=False),
			*block(4, 6, normalize=False),
			nn.Linear(6, int(np.prod(self.input_shape))),
			nn.Tanh()
		)

	def forward(self, z):
		#print(z.shape)
		x = self.model(z)
		#print(x.shape)
		x = x.view(x.size(0), *self.input_shape)
		#print(x.shape)
		return x

	def sample(self, batch_size):
		# Sample noise as generator input
		z = Variable(torch.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
		#print(z.shape)
		# Generate a batch of images
		gen_experiences = self.model(z).detach().numpy()
		result = descale(torch.Tensor(gen_experiences), self.state_low, self.state_high,
		                 self.action_low, self.action_high, self.reward_low, self.reward_high)

		return (
			torch.FloatTensor(result[:, 0:3]).to(self.device),
			torch.FloatTensor(result[:, 3]).unsqueeze(1).to(self.device),
			torch.FloatTensor(result[:, 4:7]).to(self.device),
			torch.FloatTensor(result[:, -2]).unsqueeze(1).to(self.device),
			torch.FloatTensor(np.random.choice(2, batch_size, p=[1 / 200.0, 199 / 200.0])).unsqueeze(1).to(self.device)
		)


class Discriminator(nn.Module):
	def __init__(self, action_shape, state_shape, action_low, action_high, state_low, state_high, latent_dim=2):
		super(Discriminator, self).__init__()

		self.action_shape = action_shape
		self.state_shape = state_shape
		self.feature_size = self.action_shape + (2 * self.state_shape) + 1
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
			nn.Linear(int(np.prod(self.input_shape)), 6),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(6, 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(4, 1),
			nn.Sigmoid(),
		)

	def forward(self, x):
		x_flat = x.view(x.size(0), -1)
		validity = self.model(x_flat)

		return validity




