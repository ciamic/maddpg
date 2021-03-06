from collections import namedtuple, deque
import random
import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # try to use GPU if available

class ReplayBuffer:
    
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, num_agents, seed):
        
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.num_agents = num_agents
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["states", 
                                                                "actions", 
                                                                "rewards", 
                                                                "next_states", 
                                                                "dones"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        
        """Add a new experience to memory."""
        
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        
        """Randomly sample a batch of experiences from memory."""
        
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = [torch.from_numpy(np.vstack([e.states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        actions = [torch.from_numpy(np.vstack([e.actions[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        rewards = [torch.from_numpy(np.vstack([e.rewards[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        next_states = [torch.from_numpy(np.vstack([e.next_states[index] for e in experiences if e is not None])).float().to(device) for index in range(self.num_agents)]
        dones = [torch.from_numpy(np.vstack([e.dones[index] for e in experiences if e is not None]).astype(np.uint8)).float().to(device) for index in range(self.num_agents)]
        return (states,actions,rewards,next_states,dones)
        
    def __len__(self):
        
        """Return the current size of internal memory."""
        
        return len(self.memory)