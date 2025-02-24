import gymnasium as gym
import numpy as np
class KnapsackEnv(gym.env):

    def __init__(self,c,w,W_max,mean,std):
        self.c = c
        self.w = w 
        self.W_max = W_max
        self.n_items = len(w)
        self.observation_space = gym.spaces.MultiBinary(self.n_items)
        self.action_space = gym.spaces.MultiBinary(self.n_items)
        self.state = None
        self.mean = mean
        self.std = std
    def _get_obs(self):
        pass
    def reset(self,seed = None):
        super().reset(seed  = seed)

        self.state = self.observation_space.sample()

        return self.state,None
    
    def step(self,action):
        # Properly handle this
        self.state = action + self.state

        noise = np.random.normal(self.mean,self.std,self.n_items)

        # Calc stochastic weights
        # If weights > W_max
        # terminated = True
        # What should reward be? Related to sebastiens L1 relaxation?




        weights = self.w + noise
        tot_weight = np.sum(self.state * weights)
        # pos or neg reward ?
        if self.W_max > tot_weight:
            reward = tot_weight - self.W_max
            terminated = True
        else
            reward = np.sum(self.c*action)
            terminated = False
        return self.state,reward,terminated,False,None



