
from models import brute,gym,knapsack,policy
import numpy as np
import numpy as np
from models import q_table
import random



# Have no idea how this works
def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

def main():

    c = np.array([1,2,2,5,1])
    w = np.array([2,3,1,4,1])
    a = np.array([
        [0.1,0.2,0.2,0.1,0.5],
        [0.3,0.4,0.1,0.3,0.2]
    ])
    b = np.array([1,2])
    W_max = 10
    m = knapsack.Knapsack(-c,w,a,b,W_max)
    init_state = np.array([0,0,0,0,0])
    gym_model = gym.KnapsackEnv(c,w,W_max,0,0.01)
    gym_model.reset(seed = 0)
    # gym_model.state = init_state
    init_state = gym_model.state
    # node = m.get_LP_formulation()
    # solver = brute.BruteForceMILP() # Might need to redine the way the solver works

    # Init model
    # Init solver
    # Init gym
    # Init adv table


    training_iters = 1000
    replay = []
    train_every = 10
    train_batch = 50
    n_states = 2**5
    n_actions = 5
    # q = np.random.uniform(0,1,size = (n_actions,n_states))
    q = np.zeros((n_actions,n_states))
    print(q)
    # Set inital state
    print(gym_model.state)
    m.update_state(init_state)
    for i in range(training_iters):
        node = m.get_LP_formulation()
        solver = brute.BruteForceMILP(node)
        solver.solve(store_pool = True)
        pool = solver.pool
        obj_vals = np.array([sol.fun for sol in pool])
        pol = policy.policy_dist(obj_vals,beta = 0.5)
        if len(pol) < 1:
            continue
        action_i = categorical(pol)
        action = pool[action_i].x[0:-1]
        # print("objvals",obj_vals)
        # print("pol",pol)
        # print("actio_i",action_i)
        # # action = np.array(solver.sol.x[0:-1])
        # print("action",action)
        cur_state = gym_model.state
        obs,reward,terminated,_,action_number = gym_model.step(action)
        exp = (reward,action_number,cur_state,obs)
        replay.append(exp)
        if terminated:
            print(exp)
            obs,_ = gym_model.reset()
            print("terminated")
            
        m.update_state(obs)
        
        
        if i % train_every:
            for j in range(train_batch):
                exp = random.choice(replay)
                (reward,action_number,cur_state,obs) = exp
                
                # print("reward1",exp)
                # print("reward1",reward)
                
                # Apply q learning
                # grad = ...
                # adv = q_table.adv(q_table)
                non_bin_obs = obs
                non_bin_cur = cur_state
                obs = obs.astype(int)
                cur_state = cur_state.astype(int)
                if cur_state[0] == 1 and action_number == 0:
                    print("amasmdsma")
                    print(cur_state)
                    print(obs)
                q = q_table.q_update(q,[reward],0.01,0.9,[action_number],[int(''.join(map(str, cur_state)), 2)],[int(''.join(map(str, obs)), 2)])
                
    # print(np.argmax(q,axis = 0))

    for i in range(len(np.argmax(q,axis = 0))):
        print("State: ",'{0:05b}'.format(i))
        print("best action: ",np.argmax(q,axis = 0)[i])
    # [3 3 1 2 3 3 1 1 3 3 2 2 3 0 4 0 3 0 2 2 3 0 1 0 3 3 2 0 0 0 0 0]
    print(np.argmax(q,axis = 0))
    print(q)
    
    # print(q.shape)
        
        
        
        # print(obs,reward)
        
        
        # update model
        # solve model
        # Build policy
        # step gym env
        # save in replay buffer (with solver gradient?)
        # Sample replay buffer # (train every something i think)
        # Extract adv
        # update adv
        # update weights
        





if __name__ == "__main__":
    main()