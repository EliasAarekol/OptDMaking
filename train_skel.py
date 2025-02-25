
from models import brute,gym,knapsack,policy
import numpy as np
import numpy as np



# Have no idea how this works
def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)

def main():

    c =- np.array([1,2,2,5,1])
    w = np.array([2,3,1,4,1])
    a = np.array([
        [0.1,0.2,0.2,0.1,0.5],
        [0.3,0.4,0.1,0.3,0.2]
    ])
    b = np.array([1,2])
    W_max = 10
    m = knapsack.Knapsack(c,w,a,b,W_max)
    init_state = np.array([0,0,0,0,0])
    gym_model = gym.KnapsackEnv(c,w,W_max,0,0.01)
    gym_model.reset()
    # gym_model.state = init_state
    init_state = gym_model.state
    # node = m.get_LP_formulation()
    # solver = brute.BruteForceMILP() # Might need to redine the way the solver works

    # Init model
    # Init solver
    # Init gym
    # Init adv table


    training_iters = 10
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
        action_i = categorical(pol)
        action = pool[action_i].x[0:-1]
        print("objvals",obj_vals)
        print("pol",pol)
        print("actio_i",action_i)
        # action = np.array(solver.sol.x[0:-1])
        print("action",action)
        obs,reward,terminated,_,_ = gym_model.step(action)
        if terminated:
            obs,_ = gym_model.reset()
            print("terminated")
            break
        m.update_state(obs)
        
        print(obs,reward)
        
        
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