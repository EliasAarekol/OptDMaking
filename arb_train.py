
import numpy as np
from src.models import arb_bin
from src.gym_envs import arb_binary_gym_env ,arb_discrete_gym_env
from src.critic import q_table
from src.solvers import bnb
import matplotlib.pyplot as plt
from time import time

from src import actor

def main():
    prob_size = 3
    np.random.seed(5)
    num_cons = 6
    num_pieces = 2
    aA = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
    aB = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
    b = np.random.uniform(0,.1,size=(num_pieces,))
    c = - np.random.randint(0,10,size=(prob_size,))
    state = np.random.randint(2,size = prob_size)
    A = np.random.randint(0,2,size = (prob_size,prob_size))
    B = np.random.randint(0,2,size = (prob_size,prob_size))
    C = np.random.uniform(0,1,size = (num_cons,prob_size))
    D = np.random.uniform(0,2,size = (num_cons,prob_size))
    E = np.random.uniform(1,3,size = (num_cons))
    # bounds   = [(0,1) for _ in range(len(c))]
    # bounds = [tuple(sorted((np.random.randint(-10,10),np.random.randint(0,20)))) for _ in range(len(c))]
    bounds = [(0,10) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    m = arb_bin.Arbbin(c,C,D,E,aA,aB,b,bounds,integer)
    gym_model = arb_discrete_gym_env.Arb_binary(c,np.zeros_like(c),A,B,C,D,E,1)

    n_states = 999
    n_actions = 999
    

    lr = .01
    df = .9
    beta = .5

    m.update_state(gym_model.reset()[0])
    solver = bnb.BranchAndBoundRevamped()
    q = np.zeros((n_actions,n_states))
    critic = q_table.Q_table(q,lr,df,eps = .01)
    state = gym_model.state
    act = actor.Actor(m,solver,critic,beta = beta,lr = lr,df = df)
    # act.init_q_table(q)


    model_values = {}

    training_iters = 200
    rollout_iters = 10
    total_iters = 4000
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
            store = True
            if action is None:
                action = np.zeros_like(state)
                store = False
            obs,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            print(reward,old_state,new_state,action)

            if store:
                act.update_buffers(reward,action_number,old_state,new_state)
            model_values[new_state] = act.value_est
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
        # q = act.q_table
        # print(m.w)
    print(model_values)
    print("Training took: ",time()-start,"seconds")
    ax = plt.subplot(3,1,1)
    ax.set_title("Episodic reward")
    plt.plot(range(len(ep_rewards)),ep_rewards)
    ax = plt.subplot(3,1,2)
    ax.set_title("Mean episodic reward per policy")
    plt.plot(range(len(ep_rewards_per_p)),ep_rewards_per_p)

    ax = plt.subplot(3,1,3)
    ax.set_title("Episodic reward moving average (w = 100)")

    plt.plot(range(len(ep_rewards)-99),moving_average(ep_rewards,100))
    # plt.subplot(4,1,4)
    # plt.imshow(q, cmap='hot', interpolation='nearest')
    plt.show()
    plt.savefig("res.png")
    

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    main()
    # print(timeit.timeit(main),number = 1)