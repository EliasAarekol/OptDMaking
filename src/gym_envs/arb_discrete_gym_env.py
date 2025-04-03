import gymnasium as gym
import numpy as np
from itertools import product
class Arb_binary(gym.Env):

    def __init__(self,c,p,A,B,C,D,E,pf):
        self.c = c 
        self.A = A
        self.B = B 
        self.C = C 
        self.D = D 
        self.E = E 
        self.pf = pf
        self.p = p
        self.observation_space = gym.spaces.MultiDiscrete(9 * np.ones((A.shape[1],)))
        self.action_space = gym.spaces.MultiDiscrete(100 * np.ones((A.shape[1],)))
        self.state = None
        
    def _get_obs(self):
        pass
    def reset(self,seed = None):
        super().reset(seed  = seed)

        # self.state = self.observation_space.sample(mask = np.zeros((self.A.shape[1],)).astype(np.int8))
        self.state = np.zeros((self.A.shape[1]))

        return self.state,None
    
    def step(self,action):
        if not self.action_space.contains(action):
            print(action)
            print(self.A.shape)
            raise Exception("Action does not belong to action space")
        # Noise = ...
        nxt_state = self.A @ self.state + self.B @ action # + noise
        slack =  self.C @ self.state + self.D @ action - self.E
        
        reward = self.c @ action + self.p @ nxt_state
        terminated = False
        # print(slack)
        slack_terminated = False
        if np.any(slack >= 0):
            slack_terminated = True
            reward = -np.sum(slack[slack > 0]*self.pf)
            if reward > 0:
                raise Exception()
        for comb in product(range(10),repeat = 3):
            # print(comb)
            terminated = True
            if np.all(self.C @ nxt_state + self.D @ np.array(comb) <= self.E):
                terminated = False or slack_terminated
                break
        # if np.all(np.logical_not(action.astype(int))):
        #     terminated = True
        if not self.observation_space.contains(nxt_state):
            terminated = True
            # nxt_state = np.zeros((self.A.shape[1]))
        
        old_state = int(''.join(map(str, self.state.astype(int))))

        new_state = int(''.join(map(str, nxt_state.astype(int))))
        self.state = nxt_state
        action_index = int(''.join(map(str, action.astype(int))))
        
        if old_state == new_state:
            terminated = True
        info = {
        "action" : action_index,
        "old_state" : old_state,
        "new_state" : new_state
        }
        
     
        return self.state,reward,terminated,False,info
    
    def action_to_index(self, action):
        if np.sum(action) == 0:
            return len(action)
        else:
            return np.argmax(action)
    def state_to_index(self,state):
        return int(''.join(map(str, state.astype(int))), 2)
    
    




# c =- np.array([1,2,2,5,1])
# w = np.array([2,3,1,4,1])
# W_max = 10
# action = np.array([1,0,0,0,0])
# g = KnapsackEnv(c,w,W_max,0,0.1)
# g.reset()
# print(g.step(action))