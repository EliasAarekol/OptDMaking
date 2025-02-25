
from models import brute,gym,knapsack
import numpy as np

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
    # node = m.get_LP_formulation()
    # solver = brute.BruteForceMILP() # Might need to redine the way the solver works

    # Init model
    # Init solver
    # Init gym
    # Init adv table


    training_iters = 10
    # Set inital state
    m.update_state(init_state)
    for i in range(training_iters):
        node = m.get_LP_formulation()
        solver = brute.BruteForceMILP(node)
        solver.solve()
        action = np.array(solver.sol.x[0:-1])
        # print(action)
        obs,reward,terminated,_,_ = gym_model.step(action)
        if terminated:
            obs,_ = gym_model.reset()
            print("terminated")
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