import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

noise_max = 0.1
noise_min = 0
noi_decay = 80000
noise_by_frame = lambda frame_idx: noise_min + (noise_max - noise_min) * math.exp(
    -1. * frame_idx / noi_decay)

def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
	def __init__(self, args, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action

		if args.use_orthogonal_init:
			print("------use_orthogonal_init------")
			# The neural network layer is orthogonally initialized, and the activation layer is not
			orthogonal_init(self.l1)
			orthogonal_init(self.l2)
			orthogonal_init(self.l3, gain=0.01)
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, args, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 1)

		if args.use_orthogonal_init:
			print("------use_orthogonal_init------")
			# The neural network layer is orthogonally initialized, and the activation layer is not
			orthogonal_init(self.l1)
			orthogonal_init(self.l2)
			orthogonal_init(self.l3)
			orthogonal_init(self.l4)
			orthogonal_init(self.l5)
			orthogonal_init(self.l6)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		q2 = F.relu(self.l4(sa))
		q2 = F.relu(self.l5(q2))
		q2 = self.l6(q2)
		return q1, q2


	def Q1(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class TD3(object):
	def __init__(self, args, state_dim, action_dim, max_train_steps, max_action):
		self.lr_a = 3e-4
		self.lr_c = 3e-4
		self.use_lr_decay = args.use_lr_decay		# Learning Rate Decay
		self.use_grad_clip = args.use_grad_clip		# Gradient clip
		self.set_adam_eps = args.set_adam_eps		# Set Adam epsilon=1e-5
		self.alpha = 2.5

		self.actor = Actor(args, state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.critic = Critic(args, state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)

		if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
			self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
		else:
			self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
			self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

		self.max_action = max_action
		self.discount = args.discount
		self.tau = args.tau
		self.policy_noise = args.policy_noise
		self.noise_clip = args.noise_clip
		self.policy_freq = args.policy_freq

		self.total_it = 0
		self.max_train_steps = max_train_steps


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, total_steps, batch_size=256):
		self.total_it += 1
		policy_noise = noise_by_frame(total_steps)
		# Sample replay buffer 
		state, action, next_state, reward = replay_buffer.sample(batch_size)

		with torch.no_grad():
			next_action = (self.actor_target(next_state))

			# Compute the target Q value
			target_Q1, target_Q2 = self.critic_target(next_state, next_action)
			target_Q = torch.min(target_Q1, target_Q2)
			target_Q = reward + self.discount * target_Q

		# Get current Q estimates
		current_Q1, current_Q2 = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		if self.use_grad_clip:  # Trick 7: Gradient clip
			torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
		self.critic_optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

			pi = self.actor(state)
			Q = self.critic.Q1(state, pi)
			lmbda = self.alpha/Q.abs().mean().detach()

			actor_loss = -lmbda * Q.mean() + F.mse_loss(pi, action)	# Behavioral Cloning
			
			# Optimize the actor 
			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			if self.use_grad_clip:  # Trick 7: Gradient clip
				torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
			self.actor_optimizer.step()

			# Update the frozen target models
			for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		if self.use_lr_decay:  # 6:learning rate Decay
			self.lr_decay(total_steps)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)


	def lr_decay(self, total_steps):
		lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
		lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
		for p in self.actor_optimizer.param_groups:
			p['lr'] = lr_a_now
		for p in self.critic_optimizer.param_groups:
			p['lr'] = lr_c_now
		