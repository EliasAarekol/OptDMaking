import numpy as np
from src.solvers.bnb import BranchAndBoundRevamped
from gg1test import naive_branch_sample
from src.utils.policy import policy_dist,nabla_log_pi, categorical




if __name__ == '__main__':

    prob_size = 5
    np.random.seed(2)
    c = -np.random.randint(0,10,size=(prob_size,))
    # a = np.random.uniform(0,1,size = (2,2))
    # # b = np.random.uniform(0,1,size=(2,))
    # a = np.zeros((2,prob_size))
    # b = np.zeros((2,))
    # wx = np.random.uniform(1,2,size = (prob_size,))
    # W_x_max = np.random.randint(20,30,size = (1,) )
    bounds = [(0,20) for _ in range(len(c))]
    integer = np.ones_like(c)
    A_ub = np.random.uniform(0,2,size = (2,prob_size))     # [[0.8473096  1.29178823 0.87517442 1.783546   1.92732552]
                                                          #  [0.76688304 1.58345008 1.05778984 1.13608912 1.85119328]]
    b_ub = np.random.uniform(5,10,size = (2,))     # [5.35518029 5.4356465 ]

# This one is good: [[0.64107287 0.30885335 1.39772538 0.23990109 0.97035182]
#  [1.26547546 1.63645344 1.36605199 0.99712234 1.17359396]]
# [8.59877116 6.29249043] (numpy seed = 2)
    print(A_ub)
    print(b_ub)

    node = {
        'c' : c,
        'A_ub' : A_ub,
        'A_eq' : None,
        'b_eq' : None,
        'b_ub' : b_ub,
        'integer' : integer,
        'bounds' : bounds
    }    

    solver = BranchAndBoundRevamped()
    sols = solver.solve(node,verbose = False)
    print(sols)
    print(len([sol for sol in sols if sol["fathomed"]]))
    print(len([sol for sol in sols if not sol["fathomed"]]))
    obj_values =np.array( [sol["fun"] if sol["fathomed"] else float('inf') for sol in sols ])
    pol = policy_dist(obj_values,1)
    print(pol)
    draw = categorical(pol)
    print(draw)
    print(sols[draw])
    if sols[draw]["fathomed"]:
        sample = naive_branch_sample(sols[draw]["conds"],len(c),bounds)
        print(sample)
    



    c = np.random.randint(0,10,size=(prob_size,))
    state = np.random.randint(2,size = prob_size)
    C = np.random.uniform(0,1,size = (prob_size,prob_size))
    D = np.random.uniform(0,2,size = (prob_size,prob_size))
    E = np.random.uniform(0,2,size = (prob_size))
    rhs = E - C @ state
    lhs = C
    bounds   = [(0,1) for _ in range(len(c))]
    integer = [1 for _ in range(len(c))]
    node = {
        'c' : c,
        'A_ub' : A_ub,
        'A_eq' : None,
        'b_eq' : None,
        'b_ub' : b_ub,
        'integer' : integer,
        'bounds' : bounds
    }    

    solver = BranchAndBoundRevamped()
    sols = solver.solve(node,verbose = False)
    print(len(sols))
    print(sols[0]['x'])