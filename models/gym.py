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
        self.observation_space.seed(seed=seed)
        self.state = self.observation_space.sample(mask = np.array([0,0,0,0,0],dtype=np.int8))
        weights_left = self.w*(1-self.state.astype(int))
        while np.all(self.state.astype(int)):
            self.state = self.observation_space.sample()
        # print(self.state)

  
        # self.state = np.zeros((self.n_items,))

        return self.state,None
    
    def step(self,action):
        if not self.action_space.contains(action):
            print(action)
            raise Exception("Action does not belong to action space")
        # Properly handle this

        # if self.state[0] == 1 and np.argmax(action) == 0:
        #     print("whaaaat")
        #     print(action)
        #     print(self.state)
            
        old_state = self.state_to_index(self.state)
        self.state = action + self.state
        new_state = self.state_to_index(self.state)
        # if old_state == 7:
        #     print(self.state)
        #     print(action)
        #     weights = self.w

        #     tot_weight = np.sum(self.state * weights)
        #     remaining_weight = self.W_max - tot_weight

        #     print(remaining_weight)
        #     print(  self.w*(1-self.state.astype(int)))
        
        noise = np.random.normal(self.mean,self.std,self.n_items)

        # Calc stochastic weights
        # If weights > W_max
        # terminated = True
        # What should reward be? Related to sebastiens L1 relaxation?




        weights = self.w + noise
        tot_weight = np.sum(self.state * weights)
        
        remaining_weight = self.W_max - tot_weight
        
        action_index = self.action_to_index(action)
        
        # pos or neg reward ?
        if not self.observation_space.contains(self.state):
            reward = -np.sum(self.c*action)
            terminated = True
        elif self.W_max < tot_weight:
            
            
            reward = -tot_weight + self.W_max
            terminated = True
        elif action_index == len(action):
            # print(self.state)
            reward = 0
            terminated = True
        else:
            
            reward = np.sum(self.c*action)
            
            terminated = False
            weights_left = self.w*(1-self.state.astype(int))
            if np.all(weights_left[weights_left != 0] >= remaining_weight):
                terminated = True
                # print("terminated")
                
        
        info = {
            "action" : action_index,
            "old_state" : old_state,
            "new_state" : new_state
        }
        
        # if reward == 0:
        #     print("self.state",self.state)
        #     print("self.action",action)
        
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