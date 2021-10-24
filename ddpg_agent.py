import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

from replay_buffer import ReplayBuffer
from noise import OUNoise

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.09              # for soft update of target parameters
LR_ACTOR = 0.001        # learning rate of the actor 
LR_CRITIC = 0.001       # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARN_EVERY = 4         # Learn every n steps
LEARN_HOWMANY = 5       # Learn how much every time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # try to use GPU if available

class DDPGAgent():
    
    """Interacts with and learns from the environment."""
    
    def __init__(self, agent_index, state_size, action_size, full_state_size, full_action_size, num_agents, random_seed):
        
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_step = 0
        self.index = agent_index
        self.num_agents = num_agents

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        
        self.critic_local = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_target = Critic(full_state_size, full_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, num_agents, random_seed)
                
    def step(self, state, action, reward, next_state, done, logger):
    
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        self.num_step += 1
    
        # Save experience / reward
        self.memory.add(state, 
                        action, 
                        reward, 
                        next_state, 
                        done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            if self.num_step % LEARN_EVERY == 0:
                for i in range(LEARN_HOWMANY):
                    experiences = self.memory.sample()
                    self.learn(experiences, GAMMA, logger)

    def act(self, state, add_noise=True):
    
        """Returns actions for given state as per current policy."""
    
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise_sample = self.noise.sample()
            action += noise_sample
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, logger):
        
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            logger (Tensorboard SummaryWriter): logger
        """
        
        states, actions, rewards, next_states, dones = experiences
        
        tensor_states = torch.cat(states, dim=1).to(device)
        tensor_next_states = torch.cat(next_states, dim=1).to(device)        
        tensor_actions = torch.cat(actions, dim=1).to(device)
        
        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        next_actions = [self.actor_target(next_states[self.index]) if i == self.index else actions[i].detach() for i in range(self.num_agents)]        
        tensor_next_actions = torch.cat(next_actions, dim=1).to(device)
        
        Q_target_next = self.critic_target(tensor_next_states, tensor_next_actions)
        # Compute Q targets for current states (y_i)
        Q_target = rewards[self.index] + GAMMA * Q_target_next *(1 - dones[self.index])
        # Compute critic loss
        Q_exp = self.critic_local(tensor_states, tensor_actions)
        critic_loss = F.mse_loss(Q_exp,Q_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #        
        
        # Compute actor loss
        actions_pred = [self.actor_local(states[self.index]) if i == self.index else actions[i].detach() for i in range(self.num_agents)]                
        tensor_actions_pred = torch.cat(actions_pred, dim=1).to(device)
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss = -self.critic_local(tensor_states, tensor_actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()
        logger.add_scalars('agent%i/losses' % self.index, {'critic loss': cl, 'actor_loss': al}, self.num_step)

    def soft_update(self, local_model, target_model, tau):
        
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)