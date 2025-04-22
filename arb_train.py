
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
from scipy.sparse import lil_matrix, hstack, vstack, identity, block_diag, csr_matrix


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
    E = np.random.uniform(5,10,size = (num_cons))
    # bounds   = [(0,1) for _ in range(len(c))]
    # bounds = [tuple(sorted((np.random.randint(-10,10),np.random.randint(0,20)))) for _ in range(len(c))]
    bounds = [(0,8) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    m = arb_bin.Arbbin(-np.random.uniform(0,10,size = (prob_size,)),C,D,E,aA,aB,b,bounds,integer,config["model"]["penalty_factor"])
    gym_model = arb_discrete_gym_env.Arb_binary(c,np.zeros_like(c),A,B,C,D,E,config["gym"]["pf"])

    n_states = 999
    n_actions = 999
    

    window_size = config["plotting"]["window_size"]

    run = wandb.init(name = config["name"],mode = config["wandb_mode"],config = config)


    act_lr = config["actor"]["lr"]
    critic_lr = config["critic"]["lr"]
    df = config["critic"]["df"]
    beta = config["actor"]["beta"]

    m.update_state(gym_model.reset(0)[0])
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
    T = 4
    fathomed_counter = 0
    print(m.c)
    print(gym_model.state)
    ep_length = 0
    for _ in tqdm(range(total_iters),desc= "Total Iterations"):

        # c_agg,A_eq,b_eq,A_ub,b_ub = formulate_lp_with_initial_state(-c,A,B,C,D,E,T,state)
    
        # bounds_agg = [(0,9) for _ in range(A_ub.shape[1])]
        # integer_agg = [1 for _ in range(A_ub.shape[1])]
        # node = {
        #     "c" : c_agg,
        #     "A_ub" : A_ub,
        #     "b_ub" : b_ub,
        #     "A_eq" : A_eq,
        #     "b_eq" : b_eq,
        #     "bounds" : bounds_agg,
        #     "integer" : integer_agg,
        # }
        # sols = solver.solve(node)
        # if sols is None:
        #     expected_ep_reward = 0
        # else:    
        #     best = sols[0]
        #     for sol in sols:
        #         if sol["fun"] < best["fun"]:
        #             best = sol
        expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)
        # expected_ep_reward = 0
            # expected_ep_reward = -best["fun"]
        c_values = []
        columns = ["c1", "c2", "c3"]
        c_table = wandb.Table(columns=columns)

        for i in tqdm(range(rollout_iters),leave=False,desc= "Rollout"):
            action,act_info = act.act(state)
            ep_length += 1

            

            fathomed_counter =  fathomed_counter + 1 if act_info["fathomed"] else fathomed_counter
            nab = act_info["nab"]
            store = True
            if action is None:
                action = np.zeros_like(state)
                store = False
            state,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            # print(reward,old_state,new_state,action)
            
            if store:
                act.update_buffers(reward,action_number,old_state,new_state,nab)
            model_values[new_state] = act.value_est
            ep_reward += reward
            ep_reward_per_p+=reward
            run.log({"reward" : reward, "action" : action_number})
            if terminated or i == rollout_iters-1:
                ep_rewards.append(ep_reward)


                metric = {"ep_reward" : ep_reward , "fathomed_counter" : fathomed_counter, "expected_ep_reward" : expected_ep_reward, "ep_length" : ep_length}
                if len(ep_rewards) % window_size == 0:
                    metric["smooth_ep_reward"] = sum(ep_rewards[-window_size:])/window_size
                run.log(metric)
                ep_reward = 0
                fathomed_counter = 0
                ep_length = 0
                state,_ = gym_model.reset()
                print(state)
                expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)

                # c_agg,A_eq,b_eq,A_ub,b_ub = formulate_lp_with_initial_state(-c,A,B,C,D,E,T,state)
    
                # bounds_agg = [(0,9) for _ in range(A_ub.shape[1])]
                # integer_agg = [1 for _ in range(A_ub.shape[1])]
                # node = {
                #     "c" : c_agg,
                #     "A_ub" : A_ub,
                #     "b_ub" : b_ub,
                #     "A_eq" : A_eq,
                #     "b_eq" : b_eq,
                #     "bounds" : bounds_agg,
                #     "integer" : integer_agg,
                # }
                    
                # sols = solver.solve(node)
                # if sols is None:
                #     expected_ep_reward = 0
                # else:
                #     best = sols[0]
                #     for sol in sols:
                #         if sol["fun"] < best["fun"]:
                #             best = sol
                #     expected_ep_reward = -best["fun"]
        
                # print(obs)
            # state = obs
        ep_rewards_per_p.append(ep_reward_per_p)
        ep_reward_per_p = 0
        act.train(iters = training_iters,sample = config["actor"]["sample"],num_samples=config["actor"]["num_samples"])
        c_table.add_data(*m.c)
        run.log({"c_values" : c_table})
        c_diff = ((-c -m.c )**2).mean()
        aA_change = np.sum((aA-m.aA)**2 )
        aB_change = np.sum((aB-m.aB)**2)
        b_change = np.sum((b-m.b)**2)
        run.log({"c_diff" : c_diff , "aA_change" : aA_change , "aB_change" : aB_change, "b_change" : b_change})
        data = {}
        data["c"] = m.c.tolist()
        data["aA"] = m.aA.tolist()
        data["aB"] = m.aB.tolist()
        data["b"] = m.b.tolist()
        with open(f'params/{run.id}.yaml','w') as f:
            yaml.dump(data,f)
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
    

def formulate_lp_with_initial_state(c, A, B, D, E, F, T, s_initial):
    """
    Formulates the time-dependent LP into standard form min c'z s.t.
    A_eq z = b_eq, A_ub z <= b_ub, assuming a fixed initial state s_0.

    Args:
        c (np.ndarray): Cost vector for x_t (dim n).
        A (np.ndarray or sparse matrix): State transition matrix for s_t (dim m x m).
        B (np.ndarray or sparse matrix): State transition matrix for x_t (dim m x n).
        D (np.ndarray or sparse matrix): Inequality matrix for s_t (dim k x m).
        E (np.ndarray or sparse matrix): Inequality matrix for x_t (dim k x n).
        F (np.ndarray): Right-hand side for inequality constraints (dim k).
        T (int): Time horizon (number of steps, x_t goes from 0 to T-1).
        s_initial (np.ndarray): The fixed initial state vector s_0 (dim m).

    Returns:
        tuple: (c_agg, A_eq, b_eq, A_ub, b_ub)
               Ready for scipy.optimize.linprog (bounds need to be added separately).
               Matrices A_eq and A_ub are returned as CSR sparse matrices.
    """
    # Ensure inputs are numpy arrays for shape info
    c = np.asarray(c)
    s_initial = np.asarray(s_initial)
    F = np.asarray(F)

    # --- Dimensions ---
    n = B.shape[1]  # Dimension of x_t
    m = A.shape[0]  # Dimension of s_t
    if D is not None and E is not None:
       k = D.shape[0] # Number of inequality constraints per step
    else: # Handle case with no inequality constraints D, E, F
        k = 0


    if s_initial.shape[0] != m:
        raise ValueError(f"s_initial dimension ({s_initial.shape[0]}) must match A rows ({m})")
    if c.shape[0] != n:
        raise ValueError(f"c dimension ({c.shape[0]}) must match B columns ({n})")
    if k > 0 and F.shape[0] != k:
         raise ValueError(f"F dimension ({F.shape[0]}) must match D rows ({k})")


    N = T * n + (T + 1) * m # Total number of variables in z

    # --- Aggregated Cost Vector c_agg ---
    c_agg_x = np.tile(c, T)
    c_agg_s = np.zeros((T + 1) * m)
    c_agg = np.hstack([c_agg_x, c_agg_s])

    # --- Equality Constraints (Dynamics) A_eq_dynamics z = 0 ---
    num_eq_dynamics = T * m
    A_eq_dynamics = lil_matrix((num_eq_dynamics, N))
    I_m = identity(m, format='csr') # Use sparse identity

    for t in range(T):
        row_start = t * m
        row_end = (t + 1) * m

        col_start_xt = t * n
        col_end_xt = (t + 1) * n

        col_start_st = T * n + t * m
        col_end_st = T * n + (t + 1) * m

        col_start_st1 = T * n + (t + 1) * m
        col_end_st1 = T * n + (t + 2) * m

        A_eq_dynamics[row_start:row_end, col_start_xt:col_end_xt] = B
        A_eq_dynamics[row_start:row_end, col_start_st:col_end_st] = A
        A_eq_dynamics[row_start:row_end, col_start_st1:col_end_st1] = -I_m

    b_eq_dynamics = np.zeros(num_eq_dynamics)

    # --- Equality Constraints (Initial State) A_eq_s0 z = s_initial ---
    num_eq_s0 = m
    A_eq_s0 = lil_matrix((num_eq_s0, N))
    s0_col_start = T * n # Column index where s_0 variables begin
    s0_col_end = T * n + m
    A_eq_s0[:, s0_col_start:s0_col_end] = I_m

    b_eq_s0 = s_initial # RHS is the fixed initial state

    # --- Combine Equality Constraints ---
    A_eq = vstack([A_eq_dynamics, A_eq_s0], format='csr')
    b_eq = np.concatenate([b_eq_dynamics, b_eq_s0])

    # --- Inequality Constraints A_ub z <= b_ub ---
    if k > 0:
        num_ineq = T * k
        A_ub_x = block_diag([E] * T, format='csr') # Size (T*k) x (T*n)
        A_ub_s_main = block_diag([D] * T, format='csr') # Size (T*k) x (T*m) (for s_0 to s_{T-1})
        A_ub_s_T_zeros = csr_matrix((num_ineq, m)) # Zero block for s_T columns, Size (T*k) x m
        A_ub_s = hstack([A_ub_s_main, A_ub_s_T_zeros], format='csr') # Size (T*k) x ((T+1)*m)
        A_ub = hstack([A_ub_x, A_ub_s], format='csr') # Size (T*k) x N

        if F.ndim > 1:
            F_flat = F.flatten()
        else:
            F_flat = F
        b_ub = np.tile(F_flat, T)
    else: # No inequality constraints
        # Create empty structures as placeholders or handle as needed by solver
        A_ub = None # Or csr_matrix((0, N)) depending on solver needs
        b_ub = None # Or np.array([])

    return c_agg, A_eq, b_eq, A_ub, b_ub



def calc_expected_reward(c,A,B,C,D,E,T,state,solver):
    sols = None
    while sols is None and T > 0:

        c_agg,A_eq,b_eq,A_ub,b_ub = formulate_lp_with_initial_state(-c,A,B,C,D,E,T,state)

        bounds_agg = [(0,9) for _ in range(A_ub.shape[1])]
        integer_agg = [1 for _ in range(A_ub.shape[1])]
        node = {
            "c" : c_agg,
            "A_ub" : A_ub,
            "b_ub" : b_ub,
            "A_eq" : A_eq,
            "b_eq" : b_eq,
            "bounds" : bounds_agg,
            "integer" : integer_agg,
        }
            
        sols = solver.solve(node)
        if sols is None:
            T -= 1
        else:
            best = sols[0]
            for sol in sols:
                if sol["fun"] < best["fun"]:
                    best = sol
            return -best["fun"]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    main()
    # print(timeit.timeit(main),number = 1)
