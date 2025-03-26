import numpy as np
import torch

def categorical(p):
    return (p.cumsum(-1) >= np.random.uniform(size=p.shape[:-1])[..., None]).argmax(-1)



def policy_dist(obj_vals,beta = 1):
    exps = np.exp((-1)*beta*obj_vals)
    alpha = np.sum(exps)
    return exps/alpha

def policy_dist_torch(obj_vals,beta = 1):
    exps = torch.exp((-1)*beta*obj_vals)
    alpha = torch.sum(exps)
    return torch.divide(exps,alpha)


# This math could be off
def nabla_log_pi(action_taken_object_grad,obj_vals,obj_grads,beta = 1):
    # print("obj:",obj_vals)
    # print("grads:",np.array(obj_grads))
    exps = np.exp(-beta*obj_vals)
    alpha = np.sum(exps)
    # upper_right = np.sum(beta*exps*obj_grads)
    upper_right = beta*(exps@obj_grads)
    right_side = upper_right/alpha
    left_side = beta*action_taken_object_grad
    return right_side - left_side

# Might just make one bigger func

def policy_grad(nabla_log_pi,adv):
    return np.sum(nabla_log_pi*adv)


# obj_vals = np.random.rand(1,10)
# # obj_vals = [i for i in range(10)]
# obj_grads=  np.random.rand(1,10)

# nab = nabla_log_pi(obj_vals,obj_grads)

# adv = np.random.rand(1,10)

# print(policy_grad(nab,adv))