import numpy as np
import torch
import gym
import argparse
import os
import generative_memory
from utils import scale, descale

from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils
import TD3
import OurDDPG
import DDPG
import joblib
import torch.nn.functional as F


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward


def flatten(x):
	return to_var(x.view(x.size(0), -1))


def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


def loss_fn(recon_x, x, mu, logvar):
	BCE = F.mse_loss(recon_x, x, size_average=False)

	# see Appendix B from VAE paper:
	# Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	KLD = - 0.5 * torch.sum(1 + logvar - mu**2 -  logvar.exp())
	return BCE + KLD

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")  # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=1e4, type=int)  # Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)  # Discount factor
	parser.add_argument("--tau", default=0.005)  # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
	parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
	parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
	args = parser.parse_args()

	file_name = f"{args.policy}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")

	if args.save_model and not os.path.exists("./models"):
		os.makedirs("./models")

	env = gym.make(args.env)

	data = []

	# Set seeds
	env.seed(args.seed)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
	}

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3.TD3(**kwargs)
	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)
	elif args.policy == "DDPG":
		policy = DDPG.DDPG(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	gan_batch_size = 100
	gen_batch_size = 300
	n_epochs=10
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0]
	action_low = env.action_space.low[0]
	action_high = env.action_space.high[0]
	state_low = env.observation_space.low
	state_high = env.observation_space.high
	max_action = float(action_high)

	# Loss function
	adversarial_loss = torch.nn.BCELoss()
	n_epochs = 20
	batch_size = 512
	lr = 0.001
	b1 = 0.5
	b2 = 0.999
	n_cpu = 8
	latent_dim = 2
	# Initialize generator and discriminator
	generator = generative_memory.Generator(action_dim, state_dim, action_low, action_high, state_low, state_high)
	discriminator = generative_memory.Discriminator(action_dim, state_dim, action_low, action_high, state_low, state_high)
	optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

	generator.load_state_dict(torch.load("generator"))
	generator.eval()
	discriminator.load_state_dict(torch.load("discriminator"))
	discriminator.eval()

	Tensor = torch.FloatTensor

	train_batch = torch.zeros([gan_batch_size, 9], dtype=torch.float64)
	generative_replay_index = 0

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	gr_train_count = 0
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	epochs_count=0
	d_loss = 0
	g_loss = 0

	for t in range(int(args.max_timesteps)):

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		if generative_replay_index >= gan_batch_size:
			generative_replay_index = 0
			generative_replay_index = 0
			if t >= args.start_timesteps:
				batch = torch.cat(
					(train_batch.float(), torch.cat(generator.sample(gen_batch_size), 1).float()), 0)
			else:
				batch = train_batch
			batch = scale(batch,generator.state_low, generator.state_high,
			              generator.action_low, generator.action_high,
			              generator.reward_low, generator.reward_high)

			train_iterator = DataLoader(batch, batch_size=28, shuffle=True)
			print("Training GAN [{}]".format(epochs_count))

			for epoch in range(n_epochs):
				for i, experiences in enumerate(train_iterator):
					# Adversarial ground truths
					valid = Variable(Tensor(experiences.size(0), 1).fill_(1.0), requires_grad=False)
					fake = Variable(Tensor(experiences.size(0), 1).fill_(0.0), requires_grad=False)

					# Configure input
					real_imgs = Variable(experiences).type(Tensor)[:,:-1]

					# -----------------
					#  Train Generator
					# -----------------

					optimizer_G.zero_grad()

					# Sample noise as generator input
					z = Variable(Tensor(np.random.normal(0, 1, (real_imgs.shape[0], latent_dim))))

					# Generate a batch of images
					gen_experiences = generator(z)

					# Loss measures generator's ability to fool the discriminator
					g_loss = adversarial_loss(discriminator(gen_experiences), valid)

					g_loss.backward()
					optimizer_G.step()

					# ---------------------
					#  Train Discriminator
					# ---------------------

					optimizer_D.zero_grad()

					# Measure discriminator's ability to classify real from generated samples
					real_loss = adversarial_loss(discriminator(real_imgs), valid)
					fake_loss = adversarial_loss(discriminator(gen_experiences.detach()), fake)
					d_loss = (real_loss + fake_loss) / 2

					d_loss.backward()
					optimizer_D.step()

					d_loss=d_loss.item()
					g_loss=g_loss.item()

					batches_done = epoch * len(train_iterator) + i

			print("Epoch[{}] [D loss: {}] [G loss: {}]".format(epochs_count, d_loss, g_loss))
			epochs_count = epochs_count + 1


		train_batch[generative_replay_index] = torch.Tensor(
			np.concatenate((state, action, next_state, np.array([reward]), np.array([done_bool])), 0))
		generative_replay_index = generative_replay_index + 1


		# Store data in replay buffer
		replay_buffer.add(state, action, next_state, reward, done_bool)
		#print([state, action, next_state, reward, done_bool])
		#generative_memory.add(state, action, next_state, reward, done_bool)
		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps and t>10000:
			policy.train(generator, args.batch_size)
		if t >= args.start_timesteps and t<=10000:
			policy.train(replay_buffer, args.batch_size)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}",
			)

			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			evaluations.append(eval_policy(policy, args.env, args.seed))
			np.save(f"./results/{file_name}", evaluations)
			if args.save_model: policy.save(f"./models/{file_name}")

		if(replay_buffer.ptr==1e6-1):
			r = np.concatenate((replay_buffer.state,
									replay_buffer.action,
									replay_buffer.next_state,
									replay_buffer.reward,
									replay_buffer.not_done), axis=1)
			filename = "replay2.joblib"
			joblib.dump(r, filename)