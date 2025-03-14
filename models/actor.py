from models.brute import BruteForcePara
from models.bnb import BranchAndBound
from models.policy import policy_dist,nabla_log_pi
from models.q_table import train_q_table
import numpy as np

def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)




class Actor:
    def __init__(self,model,solver = "brute",max_iter = 10000,beta = 1,lr = 0.01,df = 0.9):
        self.model = model
        # self.solver = bruteForceSolveMILP if solver == "brute" else BranchAndBound # Fix this
        self.max_iter = 10000 
        self.beta = beta
        self.lag_grads = None
        self.lag_grad_action_drawn = None
        self.n_desc_vars = self.model.n_desc_vars
        self.nab = None
        self.buffer = ExperienceBuffer()
        self.q_table = None
        self.lr = lr
        self.df = df
        self.solver = BruteForcePara(4) # Fix this
        self.value_est = 0

    def init_q_table(self,q_table):
        self.q_table = q_table

    def act(self,new_state):
        # Compute next action
        self.model.update_state(new_state)
        node = self.model.get_LP_formulation() 
        # print("b_eq",node["b_eq"])
        # print("state",new_state)

        # solver = BruteForceMILP(node)
        # solver.solve(store_pool= True)
        # sol_pool2 = solver.pool 
        sol_pool = self.solver.bruteForceSolveMILP(node,self.max_iter) # Has to return an array of dicts that include the action x, the obj func, and marginals
        # if sol_pool != sol_pool2:
        #     raise Exception(sol_pool,sol_pool2)
        if len(sol_pool) < 2:
            raise Exception("Solution pool needs atleast 2 elements")
        obj_values =np.array( [sol["fun"] for sol in sol_pool])
        pol = policy_dist(obj_values,self.beta)
        draw = categorical(pol)
        actions = [sol["x"][0:self.n_desc_vars] for sol in sol_pool]
        # Compute model specific gradient
        ineq_margs = [np.array(sol["ineqlin"]) for sol in sol_pool] # negative signs here could be wrong
        eq_margs = [np.array(sol["eqlin"]) for sol in sol_pool] #negative signs here could be wrong
        self.value_est = sol_pool[draw]["x"][-2]
        lag_grads = [self.model.lagrange_gradient(action,new_state,eq_marg,ineq_marg) for action,ineq_marg,eq_marg in zip(actions,ineq_margs,eq_margs)]
        action = actions[draw]
        lag_grad_action_drawn = self.model.lagrange_gradient(action,new_state,eq_margs[draw],ineq_margs[draw])
        # print("lag_grads",lag_grads)
        self.nab = nabla_log_pi(lag_grad_action_drawn,obj_values,lag_grads,self.beta)
        return action
    
    def train(self,iters = 1000,sample = False,num_samples = 0.5):

        # Not sure how q_table should be trained
        size = len(self.buffer.rewards)
        for _ in range(iters):
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
            self.q_table,_ = train_q_table(self.q_table,rewards,self.lr,self.df,actions,states,nxt_states)
            # print("q_table",self.q_table)
            # print("nabs",nabs)
            pol_grad = (nabs.T @ self.q_table[actions,states])/len(rewards)
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
