
import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack, identity, kron
from src.models.arb_bin import Arbbin
from src.solvers.bnb import BranchAndBoundRevamped



def dLdx(c,A_ub,A_eq,ineq,eq,upper,lower):
    A_ub = np.array([]) if A_ub is None else A_ub
    A_eq = np.array([]) if A_eq is None else A_eq
    # return c - ineq @ A_ub - eq @ A_eq - upper + lower

    return c - ineq @ A_ub - eq @ A_eq - upper - lower





num_cons = 6
prob_size = 3
num_pieces = 2
np.random.seed(5)
aA = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
aB = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
b = np.random.uniform(0,.1,size=(num_pieces,))
c = np.random.randint(0,10,size=(prob_size,))
state = np.random.randint(2,size = prob_size)
A = np.random.randint(0,2,size = (prob_size,prob_size))
B = np.random.randint(0,2,size = (prob_size,prob_size))
D = np.random.uniform(0,1,size = (num_cons,prob_size))
E = np.random.uniform(0,2,size = (num_cons,prob_size))
F = np.random.uniform(5,10,size = (num_cons))

bounds = [(0,9) for _ in range(len(c))]
integer = [1 for _ in range(len(c))]
m = Arbbin(c,D,E,F,aA,aB,b,bounds,integer,1)
m.update_state(np.zeros_like(c))
node = m.get_LP_formulation()

solver = BranchAndBoundRevamped()

sols = solver.solve(node)

print(sols)

for sol in sols:
    print(dLdx(node["c"],node["A_ub"],node["A_eq"],sol["ineqlin"],sol["eqlin"],sol["upper"],sol["lower"]))