
import numpy as np
from src.models import arb_bin
from src.gym_envs import arb_binary_gym_env ,arb_discrete_gym_env
from src.critic import q_table
from src.solvers import bnb
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import wandb
from src import actor
import yaml


def main():
    config = yaml.safe_load(open('config.yaml'))

    prob_size = config["model"]["prob_size"]
    np.random.seed(config["numpy_seed"])
    num_cons = config["model"]["n_cons"]
    num_pieces = config["model"]["n_value_func"]
    aA = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
    aB = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
    b = np.random.uniform(0,.1,size=(num_pieces,))
    c = np.random.randint(0,10,size=(prob_size,))
    state = np.random.randint(2,size = prob_size)
    A = np.random.randint(0,2,size = (prob_size,prob_size))
    B = np.random.randint(0,2,size = (prob_size,prob_size))
    C = np.random.uniform(0,1,size = (num_cons,prob_size))
    D = np.random.uniform(0,2,size = (num_cons,prob_size))
    E = np.random.uniform(1,3,size = (num_cons))
    # bounds   = [(0,1) for _ in range(len(c))]
    # bounds = [tuple(sorted((np.random.randint(-10,10),np.random.randint(0,20)))) for _ in range(len(c))]
    bounds = [(0,9) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    m = arb_bin.Arbbin(-c,C,D,E,aA,aB,b,bounds,integer,config["model"]["penalty_factor"])
    gym_model = arb_discrete_gym_env.Arb_binary(c,np.zeros_like(c),A,B,C,D,E,config["gym"]["pf"])

    n_states = 999
    n_actions = 999
    

    window_size = config["plotting"]["window_size"]

    run = wandb.init(name = config["name"],mode = config["wandb_mode"],config = config)


    act_lr = config["actor"]["lr"]
    critic_lr = config["critic"]["lr"]
    df = config["critic"]["df"]
    beta = config["actor"]["beta"]

    m.update_state(gym_model.reset()[0])
    solver = bnb.BranchAndBoundRevamped()
    q = np.zeros((n_actions,n_states))
    critic = q_table.Q_table(q,critic_lr,df,config["critic"]["eps"])
    state = gym_model.state
    act = actor.Actor(m,solver,critic,beta = beta,lr = act_lr,df = df,nn_sample = config["actor"]["nn_sample"])
    # act.init_q_table(q)


    model_values = {}

    training_iters = config["train_iters"]
    rollout_iters = config["rollout_iters"]
    total_iters = config["total_iters"]
    # q = np.random.uniform(0,1,size = (n_actions,n_states))
    # print(q)
    # Set inital state
    ep_reward =0
    ep_rewards = []
    ep_reward_per_p = 0
    ep_rewards_per_p = []
    start = time()
    # m.update_state(init_state)
    fathomed_counter = 0
    for _ in tqdm(range(total_iters),desc= "Total Iterations"):
        for i in tqdm(range(rollout_iters),leave=False,desc= "Rollout"):
            action,act_info = act.act(state)
            fathomed_counter =  fathomed_counter + 1 if act_info["fathomed"] else fathomed_counter
            store = True
            if action is None:
                action = np.zeros_like(state)
                store = False
            obs,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            # print(reward,old_state,new_state,action)
            
            if store:
                act.update_buffers(reward,action_number,old_state,new_state)
            model_values[new_state] = act.value_est
            ep_reward += reward
            ep_reward_per_p+=reward
            run.log({"reward" : reward})
            if terminated or i == rollout_iters-1:
                ep_rewards.append(ep_reward)

                metric = {"ep_reward" : ep_reward , "fathomed_counter" : fathomed_counter}
                if len(ep_rewards) % window_size == 0:
                    metric["smooth_ep_reward"] = sum(ep_rewards[-window_size:])/window_size
                run.log(metric)
                ep_reward = 0
                fathomed_counter = 0
                obs,_ = gym_model.reset()
            state = obs
        ep_rewards_per_p.append(ep_reward_per_p)
        ep_reward_per_p = 0
        act.train(iters = training_iters,sample = config["actor"]["sample"],num_samples=config["actor"]["num_samples"])
        c_diff = ((-c -m.c )**2).mean()
        aA_change = np.sum((aA-m.aA)**2 )
        aB_change = np.sum((aB-m.aB)**2)
        b_change = np.sum((b-m.b)**2)
        run.log({"c_diff" : c_diff , "aA_change" : aA_change , "aB_change" : aB_change, "b_change" : b_change})
        # q = act.q_table
        # print(m.w)
    print(model_values)
    # print("Training took: ",time()-start,"seconds")
    # ax = plt.subplot(3,1,1)
    # ax.set_title("Episodic reward")
    # plt.plot(range(len(ep_rewards)),ep_rewards)
    # ax = plt.subplot(3,1,2)
    # ax.set_title("Mean episodic reward per policy")
    # plt.plot(range(len(ep_rewards_per_p)),ep_rewards_per_p)

    # ax = plt.subplot(3,1,3)
    # ax.set_title("Episodic reward moving average (w = 100)")

    # plt.plot(range(len(ep_rewards)-(window_size-1)),moving_average(ep_rewards,window_size))
    # # plt.subplot(4,1,4)
    # # plt.imshow(q, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.savefig("res.png")
    

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    main()
    # print(timeit.timeit(main),number = 1)
