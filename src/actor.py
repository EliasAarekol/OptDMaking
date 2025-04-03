from src.utils.policy import policy_dist,nabla_log_pi,categorical,naive_branch_sample,nn_branch_sample,nn_branch_sample_only_keep_ints,naive_branch_sample_only_keep_ints
from src.utils.q_table import train_q_table
from tqdm import tqdm
import numpy as np

# def categorical(p):
#     return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)




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
        if sol_pool is None:
            return None
        # if len(sol_pool) < 2:
        #     raise Exception("Solution pool needs atleast 2 elements")
        obj_values =np.array( [sol["fun"] for sol in sol_pool])
        pol = policy_dist(obj_values,self.beta)
        draw = categorical(pol)
        chosen_sol = sol_pool[draw]
        # Sample unexplored nodes
        if chosen_sol["fathomed"]:
            if self.nn_sample:
                action = nn_branch_sample_only_keep_ints(chosen_sol['x'][self.desc_vars],chosen_sol["conds"],len(chosen_sol['x'][self.desc_vars]),node["bounds"][self.desc_vars]) # this should be edited to be more general
            else:
                action = naive_branch_sample_only_keep_ints(chosen_sol['x'][self.desc_vars],chosen_sol["conds"],len(chosen_sol['x'][self.desc_vars]),node["bounds"][self.desc_vars]) # this should be edited to be more general
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
        # Compute policy sensitivity
        self.nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        info = {
            "fathomed" : chosen_sol["fathomed"]
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





    def update_buffers(self,reward,action,state,new_state):
        self.buffer.rewards.append(reward)
        self.buffer.actions.append(action)
        self.buffer.states.append(state)
        self.buffer.nxt_states.append(new_state)
        self.buffer.nabs.append(self.nab)
        # Make sure we dont use it twice :)
        del self.nab






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
