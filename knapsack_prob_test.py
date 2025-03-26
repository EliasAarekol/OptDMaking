import numpy as np
from src.models.knapsack import Knapsack
from src.solvers.bnb import BranchAndBoundRevamped


if __name__ == '__main__':
    c = np.array([10,1])
    state = np.zeros_like(c)
    w_true= np.array([9,5])
    np.random.seed(0)
    # w = w_true + np.random.uniform(-0.5,0.5,size=(w_true.shape))
    w = np.array([0,0])
    # a = np.array([
    #     [0.1,0.2,0.2,0.1,0.5],
    #     [0.3,0.4,0.1,0.3,0.2]
    # ])
    # b = np.array([1,2])
    a = np.array([
        [0.,0.],
        [0.,0.]
    ])
    b = np.array([0,0])
    W_max = [10]
    penalty_factor = 0.1
    m = Knapsack(-c,w,a,b,W_max,p_f = penalty_factor)
    m.update_state(state)
    node = m.get_LP_formulation()
    solver = BranchAndBoundRevamped()
    sols = solver.solve(node,verbose = False)