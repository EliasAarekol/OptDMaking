
import numpy as np
from models import knapsack, gym,actor
import matplotlib.pyplot as plt
from time import time

def main():
    n_non_relevant_solution = 2
    c = np.array([1,2,2,5,1])
    w_true = np.array([2,3,1,4,1])
    # w= np.array([2,3,1,4,1])
    np.random.seed(0)
    w = w_true + np.random.uniform(-0.5,0.5,size=(5,))
    print("w:",w)
    # a = np.array([
    #     [0.1,0.2,0.2,0.1,0.5],
    #     [0.3,0.4,0.1,0.3,0.2]
    # ])
    # b = np.array([1,2])
    a = np.array([
        [0.,0.,0.,0.,0.],
        [0.,0.,0.,0.,0.]
    ])
    b = np.array([0,0])
    W_max = [10]
    m = knapsack.Knapsack(-c,w,a,b,W_max)
    init_state = np.array([0,0,0,0,0])
    gym_model = gym.KnapsackEnv(c,w_true,W_max,0,0.0)
    gym_model.reset(seed = 0)

    n_states = 2**w_true.shape[0]
    n_actions = w_true.shape[0] + 1
    
    q = np.zeros((n_actions,n_states))
    state = gym_model.state
    act = actor.Actor(m,"brute",beta = 1,lr = .1, df = 0.9)
    act.init_q_table(q)




    training_iters = 10
    rollout_iters = 20
    total_iters = 2000
    # q = np.random.uniform(0,1,size = (n_actions,n_states))
    # print(q)
    # Set inital state
    ep_reward =0
    ep_rewards = []
    ep_reward_per_p = 0
    ep_rewards_per_p = []
    start = time()
    # m.update_state(init_state)
    for _ in range(total_iters):
        for _ in range(rollout_iters):
            action = act.act(state)
            obs,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            act.update_buffers(reward,action_number,old_state,new_state)
            ep_reward += reward
            ep_reward_per_p+=reward
            if terminated:
                ep_rewards.append(ep_reward)
                ep_reward = 0
                obs,_ = gym_model.reset()
            state = obs
        ep_rewards_per_p.append(ep_reward_per_p)
        ep_reward_per_p = 0
        act.train(iters = training_iters,sample = True,num_samples=0.5)
        q = act.q_table
        # print(m.w)
    print(w_true)
    print(m.w)
    print(m.a)
    print(m.b)
    print("Training took: ",time()-start,"seconds")
    plt.subplot(3,1,1)
    plt.plot(range(len(ep_rewards)),ep_rewards)
    plt.subplot(3,1,2)
    plt.plot(range(len(ep_rewards_per_p)),ep_rewards_per_p)
    plt.subplot(3,1,3)
    plt.imshow(q, cmap='hot', interpolation='nearest')


    plt.show()
   

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    main()
    # print(timeit.timeit(main),number = 1)