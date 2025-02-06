from scipy.optimize import linprog
from collections import deque
import numpy as np
from copy import deepcopy,copy
from node import Node
from math import floor
class BranchAndBound:
    def __init__(self,init_node,sense):
        self.init_node = init_node
        self.integer = self.init_node.integer
        self.tree = []
        self.candidates = []
        self.queue = deque()
        self.sol = None


    def optimize_node(self,node):
        return linprog(
            node.c,
            node.A_ub,
            node.b_ub,
            node.A_eq,
            node.b_eq,
            node.bounds,
            )

# Handle infeasible sub sol
# Add parent and child relation
        
    def solve(self):
        res = self.optimize_node(self.init_node)
        if not res.success:
            return 
        self.tree.append(self.init_node)
        if self.all_integer(res.x):
            print(res.x[0].is_integer())
            self.sol = res
            # print("hello")
            return
        
        sol_rounded = np.floor(res.x) # Does this work for negative solutions?
        print(sol_rounded)
        print(res)
        ub = self.init_node.c @ sol_rounded
        branch_var = np.argmax(np.abs(res.x) - np.floor(np.abs(res.x)))
        print(ub)
        print(np.abs(res.x) - np.floor(np.abs(res.x)))

        branch_bound = copy(self.init_node.bounds[branch_var])
        left_branch = deepcopy(self.init_node)
        right_branch = deepcopy(self.init_node)
        left_branch.bounds[branch_var] = (left_branch.bounds[branch_var][0],floor(res.x[branch_var]))
        right_branch.bounds[branch_var] = (np.ceil(res.x[branch_var]),left_branch.bounds[branch_var][1] )
        self.queue.append(left_branch)
        self.queue.append(right_branch)

        while len(self.queue) > 0:
            node = self.queue.popleft()
            self.tree.append(node)
            res = self.optimize_node(node)
            print(res)
            print(node.bounds)
            if not res.success:
                print("infeasible")
                continue

            if self.all_integer(res.x):
                self.candidates.append(node)
                if res.fun <= ub:
                    ub = res.fun
                    self.sol = res
            elif res.fun <= ub:
                print("hello")
                branch_var = np.argmax(np.abs(res.x) - np.floor(np.abs(res.x)))

                left_branch = deepcopy(node)
                right_branch = deepcopy(node)
                left_branch.bounds[branch_var] = (left_branch.bounds[branch_var][0],np.floor(res.x[branch_var]))
                right_branch.bounds[branch_var] = (np.ceil(res.x[branch_var]),left_branch.bounds[branch_var][1] )
                self.queue.append(left_branch)
                self.queue.append(right_branch)


    def all_integer(self,x):
       return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])




# c = np.array([-5,-6])

# A_ub = np.array([[1 , 1],[4,7]])
# b_ub = np.array([5,28])
# bounds = [(0,None),(0,None)]
# integer  = [1,1]

# init_node = Node(c,A_ub=A_ub,b_ub=b_ub,bounds=bounds,integer=integer)

# solver = BranchAndBound(init_node,None)
# solver.solve()
# print(solver.sol)



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

init_node = Node(c,A_ub=A,b_ub=b,bounds=bounds,integer=integer)

solver = BranchAndBound(init_node,None)
solver.solve()
print(solver.tree)

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