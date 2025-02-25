import gymnasium as gym
import numpy as np
class KnapsackEnv(gym.Env):

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

        # self.state = self.observation_space.sample()
        self.state = np.zeros((self.n_items,))

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
        if np.any(self.state > 1):
            reward = - np.sum(self.c*action)
            print("1",reward)
            terminated = True
        elif self.W_max < tot_weight:
            
            
            reward = tot_weight - self.W_max
            print("2",reward)
            terminated = True
        else:
            
            reward = np.sum(self.c*action)
            print("3",reward)
            
            terminated = False
        return self.state,reward,terminated,False,None
    
    




# c =- np.array([1,2,2,5,1])
# w = np.array([2,3,1,4,1])
# W_max = 10
# action = np.array([1,0,0,0,0])
# g = KnapsackEnv(c,w,W_max,0,0.1)
# g.reset(0)
# print(g.step(action))