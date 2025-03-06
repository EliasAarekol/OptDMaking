
import numpy as np
from models import knapsack, gym,actor

def main():
    n_non_relevant_solution = 2
    c = np.array([1,2,2,5,1])
    w_true = np.array([2,3,1,4,1])
    # w= np.array([2,3,1,4,1])
    np.random.seed(0)
    w = w_true + np.random.uniform(-0.5,0.5,size=(5,))
    print("w:",w)
    a = np.array([
        [0.1,0.2,0.2,0.1,0.5],
        [0.3,0.4,0.1,0.3,0.2]
    ])
    b = np.array([1,2])
    # a = np.array([
    #     [0.,0.,0.,0.,0.],
    #     [0.,0.,0.,0.,0.]
    # ])
    # b = np.array([0,0])
    W_max = [10]
    m = knapsack.Knapsack(-c,w,a,b,W_max)
    init_state = np.array([0,0,0,0,0])
    gym_model = gym.KnapsackEnv(c,w_true,W_max,0,0.0)
    gym_model.reset(seed = 0)

    n_states = 2**w_true.shape[0]
    n_actions = w_true.shape[0] + 1
    
    q = np.zeros((n_actions,n_states))
    state = gym_model.state
    act = actor.Actor(m,"brute",beta = 1,lr = 0.1, df = 0.9)
    act.init_q_table(q)




    training_iters = 10
    rollout_iters = 100
    total_iters = 10
    # q = np.random.uniform(0,1,size = (n_actions,n_states))
    # print(q)
    # Set inital state
    ep_reward =0
    ep_rewards = []
    
    # m.update_state(init_state)
    for _ in range(total_iters):
        for _ in range(rollout_iters):
            action = act.act(state)
            obs,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            act.update_buffers(reward,action_number,old_state,new_state)
            if terminated:
                obs,_ = gym_model.reset()
            state = obs
        act.train(iters = training_iters,sample = False)
        print(m.w)
    print(w_true)
    print(m.w)
   



if __name__ == "__main__":
    main()
    # print(timeit.timeit(main),number = 1)