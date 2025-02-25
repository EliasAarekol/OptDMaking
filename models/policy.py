import numpy as np


def policy_dist(obj_vals,beta = 1):
    exps = np.exp(-beta*obj_vals)
    alpha = np.sum(exps)
    return exps/alpha


# This math could be off
def nabla_log_pi(obj_vals,obj_grads,beta = 1):
    exps = np.exp(-beta*obj_vals)
    alpha = np.sum(exps)
    upper_right = np.sum(beta*exps*obj_grads)
    right_side = upper_right/alpha
    left_side = beta*obj_grads
    return right_side - left_side

# Might just make one bigger func

def policy_grad(nabla_log_pi,adv):
    return np.sum(nabla_log_pi*adv)


obj_vals = np.random.rand(1,10)
# obj_vals = [i for i in range(10)]
obj_grads=  np.random.rand(1,10)

nab = nabla_log_pi(obj_vals,obj_grads)

adv = np.random.rand(1,10)

print(policy_grad(nab,adv))