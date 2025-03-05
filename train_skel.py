
from models import brute,gym,knapsack,policy
import numpy as np
import numpy as np
from models import q_table
import random
import matplotlib.pyplot as plt
import time



# Have no idea how this works
def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)




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
    W_max = 10
    m = knapsack.Knapsack(-c,w,a,b,W_max)
    init_state = np.array([0,0,0,0,0])
    gym_model = gym.KnapsackEnv(c,w_true,W_max,0,0.0)
    gym_model.reset(seed = 0)
    
    init_state = gym_model.state

    # Init model
    # Init solver
    # Init gym
    # Init adv table


    training_iters = 1000
    train_every = 10
    train_batch = 100
    # training_iters = 300
    # train_every = 2
    # train_batch = 100
    replay = []
    n_states = 2**w_true.shape[0]
    n_actions = w_true.shape[0] + 1
    # q = np.random.uniform(0,1,size = (n_actions,n_states))
    q = np.zeros((n_actions,n_states))
    # print(q)
    # Set inital state
    ep_reward =0
    ep_rewards = []
    
    m.update_state(init_state)
    for i in range(training_iters):
        # Extract action
        start = time.time()
        node = m.get_LP_formulation()
        solver = brute.BruteForceMILP(node)
        solver.solve(store_pool = True,verbose = True)
        print(time.time()-start)
        break
        pool = solver.pool
        obj_vals = np.array([sol.fun for sol in pool])
        pol = policy.policy_dist(obj_vals,beta = 0.5)

        action_i = categorical(pol)
        action = pool[action_i].x[0:-2]
        ineq_margs = [sol.ineqlin.marginals for sol in pool]
        eq_margs = [sol.eqlin.marginals for sol in pool]
        # print(eq_margs)
        sols = [sol.x for sol in pool]
        funs = np.array([sol.fun for sol in pool])

        lag_grads = [m.lagrange_gradient(x[:-n_non_relevant_solution],gym_model.state,eq_marg,ineq_marg) for x,ineq_marg,eq_marg in zip(sols,ineq_margs,eq_margs)]
        lag_grad_action_taken = m.lagrange_gradient(action,gym_model.state,pool[action_i].eqlin.marginals,pool[action_i].ineqlin.marginals)

        obs,reward,terminated,_,info = gym_model.step(action)
        action_number = info["action"]
        old_state = info["old_state"]
        new_state = info["new_state"]
        exp = (reward,action_number,old_state,new_state,lag_grads,lag_grad_action_taken,funs)
        ep_reward += reward
        replay.append(exp)
        if terminated:
            ep_rewards.append(ep_reward)
            ep_reward = 0
            # print(len(ep_rewards))
            # print(exp)
            obs,_ = gym_model.reset()
            # print("terminated")
            # print(reward)
        m.update_state(obs)
        
        # Train
        if i % train_every:
            pol_grad = np.zeros((17,))
            for j in range(train_batch):
                exp = random.choice(replay)
                (reward,action_number,old_state,new_state,lag_grads,lag_grad_action_taken,funs) = exp
                
                # Apply q learning
                # grad = ...
                # adv = q_table.adv(q_table)
                q = q_table.q_update(q,[reward],0.01,0.9,[action_number],[old_state],[new_state])
                adv = q_table.adv(q)
                nab = policy.nabla_log_pi(lag_grad_action_taken,funs,lag_grads,beta = 0.1)
                # print(q)
                # print(nab)
                # print(nab.shape)
                # local_adv = adv[action_number,old_state]
                # print(local_adv)
                # pol_grad += nab * local_adv
                pol_grad += nab * q[action_number,old_state]
                
            pol_grad /= train_batch
        # print(pol_grad)
        # break
        # print(m.w)
        # print(pol_grad)
            m.w -= 0.1*pol_grad[0:5]
            m.a[0] -= 0.01*pol_grad[5:10]
            m.a[1] -= 0.01*pol_grad[10:15]
            m.b[0] -= 0.01*pol_grad[15]
            m.b[1] -= 0.01*pol_grad[16]
            replay = []
    # print(ep_rewards)
    plt.plot(range(len(ep_rewards)),ep_rewards)
    plt.show()
    # print(q)
    print(m.w)
    # print(m.a)
    # print(m.b)

                
                
                
                
    # print(np.argmax(q,axis = 0))

    # for i in range(len(np.argmax(q,axis = 0))):
    #     print("State: ",'{0:05b}'.format(i))
    #     print("best action: ",np.argmax(q,axis = 0)[i])
    # # [3 3 1 2 3 3 1 1 3 3 2 2 3 0 4 0 3 0 2 2 3 0 1 0 3 3 2 0 0 0 0 0]
    # print(np.argmax(q,axis = 0))
    # print(q)
    
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
    # print(timeit.timeit(main),number = 1)