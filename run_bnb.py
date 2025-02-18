import numpy as np
from models.node import Node
from models.bnb import BranchAndBound
from models.lp_gradient import gradient

# c = np.array([-5,-6])

# A_ub = np.array([[1 , 1],[4,7]])
# b_ub = np.array([5,28])
# bounds = [(0,None),(0,None)]
# integer  = [1,1]

# init_node = Node(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,integer=integer)

# solver = BranchAndBound(init_node,None)
# solver.solve()
# print(solver.sol)


example_node = {
    "c" : None,
    "A_ub" : None,
    "b_ub" : None,
    "A_eq" : None,
    "b_eq" : None,
    "bounds" : None,
    "integer" : None,
    "parent" : None,
    "children" : None,
    "sol" : None
}





# Define problem parameters
values = np.array([10, 40, 30, 50])   # Item values
weights = np.array([5, 4, 6, 3])      # Item weights
capacity = 10                       # Knapsack capacity

n = len(values)

# Define the linear programming matrices
c = -values   # Maximize -> minimize negative value
A = [weights]  # Single inequality constraint for total weight
b = [capacity] # Knapsack capacity
bounds = [(0, 1) for _ in range(n)]  # Relaxed 0-1 constraint
integer  = [1,1,1,1]

node = {
    "c" : c,
    "A_ub" : A,
    "b_ub" : b,
    "A_eq" : None,
    "b_eq" :  None,
    "bounds" : bounds,
    "integer" : integer,
    "parent" : None,
    "children" : [],
    "sol" : None
}
# init_node = Node(c,A_ub=A,b_ub=b,bounds=bounds,integer=integer)
solver = BranchAndBound(node,None)
solver.solve()
# print(solver.tree)
print(solver.sol)

grad = gradient(solver.end_node)

print(grad)





# bounds = [(0,None),(0,2)]

# res = linprog(
#             c,
#             A_ub,
#            b_ub,
#            A_eq = None,
#            b_eq = None,
#            bounds = bounds,
#             )

# print(res)
