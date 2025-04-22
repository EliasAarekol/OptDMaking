from src.utils.policy import policy_dist,nabla_log_pi,categorical,naive_branch_sample,nn_branch_sample,nn_branch_sample_only_keep_ints,naive_branch_sample_only_keep_ints, policy_dist_torch
from src.utils.q_table import train_q_table
from tqdm import tqdm
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
import cvxpy as cp
from scipy.optimize import linprog

# def categorical(p):
#     return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)


def dLdx(c,A_ub,A_eq,ineq,eq,upper,lower):
    A_ub = np.array([]) if A_ub is None else A_ub
    A_eq = np.array([]) if A_eq is None else A_eq
    # return c - ineq @ A_ub - eq @ A_eq - upper + lower

    return c - ineq @ A_ub - eq @ A_eq - upper - lower

def calc_actual_grad(node):
    sol = linprog(
        node["c"],
        node["A_ub"],
        node["b_ub"],
        node["A_eq"],
        node["b_eq"],
        node["bounds"]
    )
    ineq = sol.ineqlin.marginals
    eq = sol.ineqlin.marginals
    return ineq,eq
 


# def dLdx_2(c,aA,aB,b,ineq,upper,lower):
#     dLdxt = c -ineq @ aB - upper - lower 
#     # return c - ineq @ A_ub - eq @ A_eq - upper + lower

#     return c - ineq @ A_ub - eq @ A_eq - upper - lower


# vals = torch.rand((4,))
# vals_np = vals.numpy()
# vals.requires_grad = True
# dphidtheta = torch.rand((4,2)) #
# dphidtheta_np = dphidtheta.numpy()
# dphidtheta.requires_grad = True



# print(vals)
# pol = policy.policy_dist_torch(vals,1) # pi
# print(pol)
# log_pol = torch.log(pol)
# log_pol[2].backward() # dpi/dphi
# print("Gradient of policy w.r.t. objective values:",vals.grad)

# # jac = torch.autograd.functional.jacobian(policy.policy_dist_torch,vals)
# # print(jac)
# nab = policy.nabla_log_pi(dphidtheta_np[2],vals_np,dphidtheta_np,beta = 1) # dpi/dtheta
# print(nab)

# grad_log_pol = vals.grad  # This is d(log π) / dvals
# expected_nabla_log_pi = grad_log_pol @ dphidtheta
# print(expected_nabla_log_pi)
# error = np.linalg.norm(nab - expected_nabla_log_pi.detach().numpy())
# print(f"Error between manual and PyTorch gradients: {error}")
# print(nab,expected_nabla_log_pi)

def check_corr_grad(obj_vals,nab,beta,lag_grads,draw): # This doesnt work with sampled nab probably because htat one gradient is wrong specfically
    obj_vals_torch = torch.tensor(obj_vals,requires_grad= True)
    pol = policy_dist_torch(obj_vals_torch,beta)
    log_pol = torch.log(pol)
    log_pol[draw].backward() # dpi/dphi
    grad_log_pol = obj_vals_torch.grad
    expected =  grad_log_pol @ lag_grads
    # print(draw)
    # print(nab - expected.detach().numpy())
    assert np.linalg.norm(nab - expected.detach().numpy()) < 1e-6


def check_with_cvxpylayers(node,bounds,lag_grad,drawn_x):
    c_b = torch.tensor(node["c"],requires_grad=True)  # Objective function

    A_ub_b = torch.tensor(node["A_ub"], requires_grad=True)
    b_ub_b = torch.tensor(node["b_ub"], requires_grad=True)

    solver_args = {
        'max_iters': 50000,  # Increase max iterations (default is often 2500)
        # 'eps': 1e-5,      # Adjust tolerance if needed (SCS default is 1e-4)
        # 'verbose': True,   # Set to True to get detailed solver output for debugging
        'solve_method' : 'ECOS'
    }
    lb_b = torch.tensor([bound[0] if bound[0] is not None else -1e3 for bound in bounds],dtype=torch.float64)
    ub_b = torch.tensor([bound[1] if bound[1] is not None else 1e3 for bound in bounds],dtype=torch.float64)
    x = cp.Variable(c_b.shape[0])
    c = cp.Parameter(c_b.shape[0])  # ✅ Change to Parameter
    A_ub = cp.Parameter(A_ub_b.shape)
    b_ub = cp.Parameter(b_ub_b.shape[0])
    lb = cp.Parameter(lb_b.shape)
    ub = cp.Parameter(ub_b.shape)

    constraints = [A_ub @ x <= b_ub, lb <= x, x <= ub]

    objective = cp.Minimize(c @ x)  # ✅ Uses c as a parameter
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()  # ✅ Now this should pass

    cvxpylayer = CvxpyLayer(problem, parameters=[c,A_ub, b_ub,lb,ub], variables=[x])
    solution, = cvxpylayer(c_b, A_ub_b, b_ub_b,lb_b,ub_b,solver_args = solver_args)
    objective_value = (c_b @ solution)
    objective_value.backward()
    # assert all(torch.abs(c_b.grad[:3] - lag_grad[:3]) < 1e-3)
    if (
            not all(torch.abs(c_b.grad[:3] - lag_grad[:3]) < 1e-3)
            or  not all(torch.abs(A_ub_b.grad[:2,:3].flatten() - lag_grad[9:15]) < 1e-3)
            or not all(torch.abs(b_ub_b.grad[:2] + lag_grad[-2:]) < 1e-3) 
        ):
        print("CVXPY check failed with")
        print("c_b_grad is: ",c_b.grad[:3])
        print("Lag_grad is: ",lag_grad[:3])
        print("x is: ", drawn_x)

    


    # # Equality constraint: x1 + x2 = 2
    # A_eq_b = torch.tensor([[1, 1]], requires_grad=True, dtype=torch.float32)
    # b_eq_b = torch.tensor([2], requires_grad=True, dtype=torch.float32)





class Actor:
    def __init__(self,model,solver,critic,beta = 1,lr = 0.01,df = 0.9,nn_sample = False):
        self.model = model
        # self.solver = bruteForceSolveMILP if solver == "brute" else BranchAndBound # Fix this

        # Should maybe be own config thing

        self.desc_vars = self.model.get_desc_var_indices()
        self.lag_grads = None
        self.lag_grad_action_drawn = None
        self.n_desc_vars = self.model.n_desc_vars
        self.nab = None
        self.buffer = ExperienceBuffer()
        # self.q_table = None
        self.lr = lr
        self.df = df
        self.beta = beta
        # self.solver = BruteForcePara(4) # Fix this
        self.solver = solver
        self.critic = critic
        self.value_est = 0
        self.nn_sample = nn_sample

    # def init_q_table(self,q_table):
    #     self.q_table = q_table

    def act(self,new_state):
        # Compute next action
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation()
        # print("b_eq",node["b_eq"])
        # print("state",new_state)

        # solver = BruteForceMILP(node)
        # solver.solve(store_pool= True)
        # sol_pool2 = solver.pool
        sol_pool = self.solver.solve(node) # Has to return an array of dicts that include the action x, the obj func, and marginals
        # if sol_pool != sol_pool2:
        #     raise Exception(sol_pool,sol_pool2)
        if len(sol_pool) < 3:
            print("asdasd")
        if sol_pool is None:
            return None
        for sol in sol_pool:
            if np.any(np.abs(dLdx(node["c"],node["A_ub"],node["A_eq"],sol["ineqlin"],sol["eqlin"],sol["upper"],sol["lower"])) > 1e-4 ):
                raise Exception("dLdx isnt 0")
        #     print(dLdx(node["c"],node["A_ub"],node["A_eq"],sol["ineqlin"],sol["eqlin"],sol["upper"],sol["lower"]))
        # if len(sol_pool) < 2:
        #     raise Exception("Solution pool needs atleast 2 elements")
        obj_values =np.array( [sol["fun"] for sol in sol_pool])
        pol = policy_dist(obj_values,self.beta)
        draw = categorical(pol)
        chosen_sol = sol_pool[draw]
        bounds = chosen_sol["bounds"]
        # Sample unexplored nodes
        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action,bounds = nn_branch_sample(chosen_sol['x'][self.desc_vars],bounds)
                # action = nn_branch_sample_only_keep_ints(chosen_sol['x'][self.desc_vars],chosen_sol["conds"],len(chosen_sol['x'][self.desc_vars]),node["bounds"][self.desc_vars]) # this should be edited to be more general
            else:
                action,bounds = naive_branch_sample(chosen_sol['x'][self.desc_vars],bounds)
                
                # action = naive_branch_sample_only_keep_ints(chosen_sol['x'][self.desc_vars],chosen_sol["conds"],len(chosen_sol['x'][self.desc_vars]),node["bounds"][self.desc_vars]) # this should be edited to be more general
                # action = naive_branch_sample(chosen_sol['x'][:self.n_desc_vars],chosen_sol["conds"],self.n_desc_vars,node["bounds"][:self.n_desc_vars]) # this should be edited to be more general

        else:
            action = chosen_sol["x"]
            action = action[self.desc_vars]



        self.value_est = sol_pool[draw]["x"][-2] # Just for value function debug


        # Compute model specific gradient
        actions = [sol["x"][self.desc_vars] for sol in sol_pool]
        ineq_margs = [np.array(sol["ineqlin"]) for sol in sol_pool] # negative signs here could be wrong
        eq_margs = [np.array(sol["eqlin"]) for sol in sol_pool] #negative signs here could be wrong
        lag_grads = [self.model.lagrange_gradient(a,new_state,eq_marg,ineq_marg) for a,ineq_marg,eq_marg in zip(actions,ineq_margs,eq_margs)]

        # Convert action solution to actual action
        lag_grad_action_drawn = self.model.lagrange_gradient(action,new_state,eq_margs[draw],ineq_margs[draw])
        lag_grads[draw] = lag_grad_action_drawn
        lag_grads = np.array(lag_grads)
        # lag_grad_action_drawn = self.model.lagrange_gradient(chosen_sol["x"][self.desc_vars],new_state,eq_margs[draw],ineq_margs[draw])
        # Compute policy sensitivity
        
        nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        check_corr_grad(obj_values,nab,self.beta,lag_grads,draw) 
        # check_with_cvxpylayers(chosen_sol["node"],bounds,lag_grad_action_drawn,action) # This is specific to the problem
        
        # Check with actual grad
        node["bounds"] = bounds
        ineq,eq = calc_actual_grad(node)
        lag_grad_action_drawn = self.model.lagrange_gradient(action,new_state,ineq,eq)
        lag_grads[draw] = lag_grad_action_drawn
        lag_grads = np.array(lag_grads)
        # lag_grad_action_drawn = self.model.lagrange_gradient(chosen_sol["x"][self.desc_vars],new_state,eq_margs[draw],ineq_margs[draw])
        # Compute policy sensitivity
        
        true_nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        
        
        info = {
            "fathomed" : chosen_sol["fathomed"],
            "nab" : true_nab,
            "n_sols" : len(sol_pool)
        }
        return action,info

    def train(self,iters = 1000,sample = False,num_samples = 0.5):

        # Not sure how q_table should be trained

        size = len(self.buffer.rewards)
        if size == 0:
            raise Exception("Buffers are empty")
        for _ in tqdm(range(iters),leave=False,desc = "Training"):
            if sample:
                # t_indexes = np.ones((size*num_samples,))
                # f_indexes = np.zeros((size*(1-num_samples),))
                # indexes = np.vstack((t_indexes,f_indexes))
                indexes = np.array(range(int(size*num_samples)))
                np.random.shuffle(indexes)
            else:
                indexes = np.array(range(size))
            indexes = indexes.astype(int)
            rewards = np.array(self.buffer.rewards)[indexes]
            actions = np.array(self.buffer.actions)[indexes]
            states = np.array(self.buffer.states)[indexes]
            nxt_states = np.array(self.buffer.nxt_states)[indexes]
            nabs = np.array(self.buffer.nabs)[indexes]
            # print(actions)
            # print(nabs)
            # self.q_table,_ = train_q_table(self.q_table,rewards,self.lr,self.df,actions,states,nxt_states)
            self.critic.train(rewards,actions,states,nxt_states)



            # self.critic.train()
            qualities = self.critic.evaluate(actions,states)





            # print("q_table",self.q_table)
            # print("nabs",nabs)
            pol_grad = (nabs.T @ qualities)/len(rewards)
            self.model.update_params(pol_grad,self.lr)
        self.buffer.reset()
        return pol_grad





    def update_buffers(self,reward,action,state,new_state,nab):
        self.buffer.rewards.append(reward)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        self.buffer.nxt_states.append(new_state)
        self.buffer.nabs.append(nab)
        # self.buffer.nabs.append(self.nab)
        # # Make sure we dont use it twice :)
        # del self.nab






class ExperienceBuffer:
    def __init__(self):
        self.rewards = []
        self.actions= []
        self.states = []
        self.nxt_states = []
        self.nabs = []

    def reset(self):
        del self.rewards[:]
        del self.actions[:]
        del self.states[:]
        del self.nxt_states[:]
        del self.nabs[:]
