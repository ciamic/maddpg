from ddpg_agent import DDPGAgent
from replay_buffer import ReplayBuffer

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # try to use GPU if available

class MADDPGAgent():

    """Orchestrates a series of DDPG Agents."""
    
    def __init__(self, num_agents, state_sizes, action_sizes, random_seed):
    
        """Initialize a MADDPGAgent object.
        Params
        ======
            num_agents (int): number of agents
            state_sizes (array of int): size of state size for each agent
            action_sizes (array of int): size of action size for each agent     
            random_seed (int): random seed
        """
                
        self.num_agents = num_agents
        
        # calculate total state size over all agents for the critic
        state_sizes_total = 0
        for value in state_sizes:
            state_sizes_total += value     
                                 
        # calculate total action size over all agents for the critic
        action_sizes_total = 0
        for value in action_sizes:
            action_sizes_total += value
        
        self.ddpg_agents = []
        for i in range(self.num_agents):
            ddpg_agent = DDPGAgent(i, state_sizes[i], action_sizes[i], state_sizes_total, action_sizes_total, num_agents, random_seed)
            self.ddpg_agents.append(ddpg_agent)

        self.num_step = 0
        
    def act(self, states, add_noise=True):
    
        """Returns actions for given states as per current policy of each agent."""
        
        actions = []
        for i in range(len(states)):
            state = states[i]
            action = self.ddpg_agents[i].act(state, add_noise)
            actions.append(action)
            
        return actions
                
    def step(self, states, actions, rewards, next_states, dones, logger):
    
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        self.num_step += 1
    
        # Save experience / reward
        for i in range(self.num_agents):            
            self.ddpg_agents[i].step(states, actions, rewards, next_states, dones, logger)
                
    def reset(self):
    
        """Reset noise process for each agent."""
    
        for i in range(self.num_agents):
            self.ddpg_agents[i].reset()
            
    def save(self):
        
        """Save the status of the agents."""
        
        for i in range(self.num_agents):
            torch.save(self.ddpg_agents[i].actor_local.state_dict(), 'actor_' + str(i) + '_solved.pth')
            torch.save(self.ddpg_agents[i].critic_local.state_dict(), 'critic_' + str(i) + '_solved.pth')
            
    def load(self):
        
        """Load the status of the agents."""
    
        for i in range(self.num_agents):
            self.ddpg_agents[i].actor_local.load_state_dict(torch.load('actor_' + str(i) + '_solved.pth'))
            self.ddpg_agents[i].critic_local.load_state_dict(torch.load('critic_' + str(i) + '_solved.pth'))