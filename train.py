
import numpy as np
from src.gym_envs import example_env
from src.models import example_model

from src.critic import gae
from src.solvers import bnb

from tqdm import tqdm
import wandb
from src import actor
import yaml
from scipy.sparse import lil_matrix, hstack, vstack, identity, block_diag, csr_matrix

def main():
    config = yaml.safe_load(open('config.yaml'))

    # Init problem
    state_size = config["model"]["state_size"]
    action_size = config["model"]["action_size"]
    np.random.seed(config["numpy_seed"])
    num_cons = config["model"]["n_cons"]
    num_pieces = config["model"]["n_value_func"]
    aA = np.random.uniform(0,0.1,size = (num_pieces,state_size))
    aB = np.random.uniform(0,0.1,size = (num_pieces,action_size))
    b = np.random.uniform(0,.1,size=(num_pieces,))
    c = np.random.uniform(0,10,size=(action_size,))
    state = np.random.randint(2,size = state_size)


    C = np.random.uniform(0,1,size = (num_cons-2,state_size))
    C = np.vstack((C,np.zeros((2,state_size))))
    D = np.random.uniform(0,1,size = (num_cons,action_size))
    E = np.random.uniform(5,15,size = (num_cons-2))
    E2 = np.random.uniform(1,10,size = (2))
    E = np.hstack((E,E2))
    action_ub = 10
    
    bounds = [(0,action_ub) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    c_model = -np.random.uniform(0,10,size = (1,)) * np.ones((action_size,))
    print(c_model)
    A = np.random.uniform(0,.1,size = (state_size,state_size))
    B = np.random.uniform(0,1,size = (state_size,action_size))

    aA =np.vstack((aA,np.random.uniform(0,0.1,size = (5,state_size)))) 
                  
    aB =np.vstack((aB,np.random.uniform(0,0.1,size = (5,action_size))) )
    # # Init solver and gym model
    b = np.hstack((b,np.random.uniform(0,.1,size=(5,))))
    load = config["load"]
    if load:
        print("loading params instead of generating random...")
        params = yaml.safe_load(open(config["load_path"]))
        aA = np.array(params['aA'])
        aB = np.array(params['aB'])
        c_model = np.array(params['c'])
        b = np.array(params['b'])
        
    
    m = example_model.Arbbin(
        c_model,C,D,E,aA,aB,b,bounds,integer,config["model"]["penalty_factor"]
        )


    gym_model = example_env.Arb_binary(c,np.zeros_like(state),A,B,C,D,E,config["gym"]["pf"],a_space_size=11,std = config["gym"]["noise_std"])
    m.update_state(gym_model.reset(0)[0])

    run = wandb.init(name = config["name"],mode = config["wandb_mode"],config = config)


    window_size = config["plotting"]["window_size"]

    act_lr = config["actor"]["lr"]
    critic_lr = config["critic"]["lr"]
    df = config["critic"]["df"]
    beta = config["actor"]["beta"]
    eps = config["critic"]["eps"]

    n_actions = action_ub*(100+10+1)+1

    dims = [state_size,128,128,1]
    critic = gae.GAE(dims,critic_lr,df,eps,0.1,config['device'])
    
    solver = bnb.BranchAndBoundRevamped()
    act = actor.Actor(m,solver,critic,beta = beta,lr = act_lr,df = df,nn_sample = config["actor"]["nn_sample"],sampled_grad = config["actor"]["sampled_grad"])
    state = gym_model.state



    training_iters = config["train_iters"]
    rollout_iters = config["rollout_iters"]
    total_iters = config["total_iters"]


    ep_reward =0
    ep_rewards = []
    rewards = []


    T = config["explicit_sol_time"]
    fathomed_counter = 0
    print(m.c)
    print(gym_model.state)
    ep_length = 0

    comp_expected = config["comp_expected"]
    comp_expected_every = config["comp_expected_every"]

    columns = ["c1", "c2", "c3"]
    c_table = wandb.Table(columns=columns)

    iter_counter = 0
    expected_ep_reward = None
    last_calced = 0
    for _ in tqdm(range(total_iters),desc= "Total Iterations"):

        if  last_calced > comp_expected_every and comp_expected:
            expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)
            last_calced = 0

        for i in tqdm(range(rollout_iters),leave=False,desc= "Rollout"):
            iter_counter += 1
            ep_length += 1
            last_calced +=1


            action,act_info = act.act(state)
            

            fathomed_counter =  fathomed_counter + 1 if act_info["fathomed"] else fathomed_counter
            nab = act_info["nab"]
            t_nab = act_info["t_nab"]
            store = True
            if action is None:
                action = np.zeros_like(state)
                store = False
            state,reward,terminated,_,info = gym_model.step(action)
            action_number = info["action"]
            old_state = info["old_state"]
            new_state = info["new_state"]
            
            if store:
                act.update_buffers(reward,action_number,old_state,new_state,nab,t_nab)

            ep_reward += reward
            run.log({"reward" : reward, "action" : action_number, "n_sols" : act_info["n_sols"]})




            if terminated or i == rollout_iters-1:
                ep_rewards.append(ep_reward)

                metric = {
                    "ep_reward" : ep_reward,
                    "fathomed_counter" : fathomed_counter,
                    "ep_length" : ep_length
                          }
                
                if len(ep_rewards) == window_size:
                    metric["smooth_ep_reward"] = sum(ep_rewards)/window_size
                    ep_rewards = []
                if comp_expected and expected_ep_reward is not None:
                    metric["expected_ep_reward"] = expected_ep_reward
                    metric["distance_from_opt_pol"] = expected_ep_reward - ep_reward
                    expected_ep_reward = None
                run.log(metric)
                ep_reward = 0
                fathomed_counter = 0
                ep_length = 0
                state,_ = gym_model.reset()
                if last_calced > comp_expected_every and comp_expected:
                    expected_ep_reward = calc_expected_reward(-c,A,B,C,D,E,T,state,solver)
                    last_calced = 0


        pol_grad = act.train(iters = training_iters,sample = config["actor"]["sample"],num_samples=config["actor"]["num_samples"])

        c_diff = ((-c -m.c )**2).mean()
        aA_change = np.sum((aA-m.aA)**2 )
        aB_change = np.sum((aB-m.aB)**2)
        b_change = np.sum((b-m.b)**2)
        run.log({"c_diff" : c_diff , "aA_change" : aA_change , "aB_change" : aB_change, "b_change" : b_change,"pol_grad": np.linalg.norm(pol_grad)})


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

        c_agg,A_eq,b_eq,A_ub,b_ub = formulate_lp_with_initial_state(c,A,B,C,D,E,T,state)

        bounds_agg = [(0,8) for _ in range(A_ub.shape[1])]
        integer_actions = [ 1 for _ in range(c.size*T)]
        integer_states = [ 0 for _ in range(A_ub.shape[1] - len(integer_actions))]
        integer_agg = integer_actions + integer_states
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
            return best["fun"]


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == "__main__":
    # pr = cProfile.Profile()
    # pr.enable()
    main()
    # pr.disable()
    # stats = Stats(pr)
    # stats.sort_stats('cumtime').print_stats(20)
    # # cProfile.run("main()k",sort = "time")
