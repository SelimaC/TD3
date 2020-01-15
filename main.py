import numpy as np
import torch
import gym
import argparse
import os
import generative_memory
import JointGAN
import datetime
import sys
import dateutil.tz
from pygit2 import Repository

from torch.utils.data import DataLoader
from torch.autograd import Variable
import utils
import TD3
import OurDDPG
import DDPG
import joblib
import torch.nn.functional as F

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


def descale(y, state_low, state_high, action_low, action_high, reward_low, reward_high, a=-1, b=1):
	x=y.clone()
	(((x[:, 0].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 1].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 2].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	(((x[:, 3].sub_(a)).div_(b-a)).mul_(action_high - action_low)).add_(action_low)
	(((x[:, 4].sub_(a)).div_(b-a)).mul_(state_high[0] - state_low[0])).add_(state_low[0])
	(((x[:, 5].sub_(a)).div_(b-a)).mul_(state_high[1] - state_low[1])).add_(state_low[1])
	(((x[:, 6].sub_(a)).div_(b-a)).mul_(state_high[2] - state_low[2])).add_(state_low[2])
	(((x[:, 7].sub_(a)).div_(b-a)).mul_(reward_high - reward_low)).add_(reward_low)
	((x[:, 8].sub_(a)).div_(b-a)).round_()


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

	print("---------------------------------------", flush=True)
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}", flush=True)
	print("---------------------------------------", flush=True)
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

	log_dir = "./logs/" + Repository('.').head.shorthand

	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

	log_file = f"{args.policy}_{args.env}_{timestamp}.txt"

	sys.stdout = open(os.path.join(log_dir, log_file), 'a+')

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
	lr = 0.0002
	b1 = 0.5
	b2 = 0.999
	batch_size = 256

	joint = JointGAN.JointGAN(action_dim, state_dim, action_low, action_high, state_low, state_high)
	# Optimizers.
	optimizer_G1 = torch.optim.Adam(joint.gxy.parameters(), lr=lr, betas=(b1, b2))
	optimizer_G2 = torch.optim.Adam(joint.gyx.parameters(), lr=lr, betas=(b1, b2))

	optimizer_D1 = torch.optim.Adam(joint.d1.parameters(), lr=lr, betas=(b1, b2))
	optimizer_D2 = torch.optim.Adam(joint.d2.parameters(), lr=lr, betas=(b1, b2))
	optimizer_D3 = torch.optim.Adam(joint.d3.parameters(), lr=lr, betas=(b1, b2))
	optimizer_D4 = torch.optim.Adam(joint.d4.parameters(), lr=lr, betas=(b1, b2))

	train_batch = torch.zeros([gan_batch_size, 9], dtype=torch.float64)
	generative_replay_index = 0
	adversarial_loss = torch.nn.BCEWithLogitsLoss()
	Tensor = torch.FloatTensor

	# Evaluate untrained policy
	evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	gr_train_count = 0
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	epochs_count=0
	d_loss = 0
	d_loss_real = 0
	d_loss_fake = 0
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
			if t >= 2000:
				#print(torch.cat(joint.sampleReplay(gen_batch_size), 1).shape)
				batch = torch.cat(
					(train_batch.float(), torch.cat(joint.sampleReplay(gen_batch_size), 1).float()), 0)
			else:
				batch = train_batch
			batch = scale(batch,joint.state_low, joint.state_high,
			              joint.action_low, joint.action_high,
			              joint.reward_low, joint.reward_high)

			idx = torch.randperm(batch.nelement())
			batch = batch.view(-1)[idx].view(batch.size())
			train_iterator = DataLoader(batch, batch_size=28, shuffle=True)
			print("Training GAN [{}]".format(epochs_count))

			for epoch in range(n_epochs):
				for i, experiences in enumerate(train_iterator):
					# Adversarial ground truths
					states, actions = Tensor(experiences[:, 0:3].float()).view(-1, state_dim), Tensor(experiences[:, 3].float()).view(
						-1, action_dim)

					valid_x = Variable(Tensor(experiences.size(0), 1).fill_(1.0), requires_grad=False)
					fake_x = Variable(Tensor(experiences.size(0), 1).fill_(0.0), requires_grad=False)

					valid_y = Variable(Tensor(experiences.size(0), 1).fill_(1.0), requires_grad=False)
					fake_y = Variable(Tensor(experiences.size(0), 1).fill_(0.0), requires_grad=False)

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
					# print(descale(experiences[:1],state_low,state_high,action_low,action_high,-20, 0))
					gen_states = joint.gxy(z1, Tensor(experiences.size(0), action_dim).fill_(0))
					# print(descale_state(gen_states[:1],state_low, state_high))
					gen_actions = joint.gyx(z2, Tensor(experiences.size(0), state_dim).fill_(0))
					# print(descale_action(gen_actions[:1], action_low,action_high))

					gen_states_from_actions = joint.gxy(z3, actions)
					# print(descale_state(gen_states_from_actions[:1],state_low,state_high))
					gen_actions_from_states = joint.gyx(z4, states)
					# print(descale_action(gen_actions_from_states[:1], action_low,action_high))

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
					# g_loss1 = adversarial_loss(joint.d_state(gen_states), valid_x)
					# g_loss2 = adversarial_loss(joint.d_action(gen_actions), valid_y)

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

					d_loss_real = real_loss1+real_loss2+real_loss3+real_loss4
					d_loss_fake = fake_loss1 + fake_loss2 + fake_loss3 + fake_loss4
					#print(descale(experiences[:1], state_low, state_high, action_low, action_high, -20, 0))

					#print(descale_state(gen_states[:1], state_low, state_high))
					#print(descale_action(gen_actions[:1], action_low, action_high))

					#print(descale_action(gen_actions_from_states[:1], action_low, action_high))
					#print(descale_state(gen_states_from_actions[:1], state_low, state_high))

					#batches_done = epoch * len(train_iterator) + i

			print("Epoch[{}] [D loss: {}] [G loss: {}]".format(epochs_count, d_loss, g_loss), flush=True)
			print("Epoch[{}] [D fake loss: {}] [D real loss: {}]".format(epochs_count, d_loss_fake, d_loss_real), flush=True)
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
		if t >= args.start_timesteps and t>20000:
			policy.train(joint, args.batch_size, True)

		if t >= args.start_timesteps and t<=20000:
			policy.train(replay_buffer, args.batch_size, False)

		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(
				f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}",flush=True
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
			r=[]
			filename = "gan_training.joblib"
			joblib.dump(r, filename)
	sys.stdout.close()