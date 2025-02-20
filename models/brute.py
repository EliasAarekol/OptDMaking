from scipy.optimize import linprog
from collections import deque
import numpy as np
from copy import copy,deepcopy
from math import floor,ceil,inf



# BFS brute force search for MILP solutions

class BruteForceMILP:
    # Implements a brute force method to solve an MILP.
    # Solves the relaxed problem and uses the rounded up solutions as an intial node for searching the solution space
    # For problems with defined bounds on variables it will test all combinations of solutions
    # For problems with no defined bounds it will test all solutions in a BFS search until the max number of iterations is completed

    def __init__(self,init_node):
        self.init_node = init_node
        self.integer = self.init_node["integer"]
        self.tree = []
        self.sol = None
        self.end_node = None
        self.queue = deque()


    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )
    
    def manhattan_round(self,x, int_indices):
        """Find the closest integer solution by rounding the integer-constrained variables."""
        x_int = x.copy()
        for i, is_int in enumerate(int_indices):
            if is_int:
                x_int[i] = round(x[i])
        return x_int

    def generate_neighbors(self,x, is_integer,bounds):
        # Needs to handle that the original bounds shouldnt be breached
        """Generate neighboring integer solutions by varying each integer-constrained variable by ±1."""
        neighbors = []
        for i, is_int in enumerate(is_integer):
            if is_int:
                l_bound,r_bound = bounds[i]
                l_bound = l_bound if l_bound is not None else -inf
                r_bound = r_bound if r_bound is not None else inf
                x_new = x.copy()
                x_new[i] = min(x[i] + 1,r_bound)
                neighbors.append(x_new)
                x_new = x.copy()
                x_new[i] = max(x[i] - 1,l_bound)
                neighbors.append(x_new)
        return neighbors

    def solve(self,max_iter = 10000,store_tree = False,verbose = False):
        res = self.optimize_node(self.init_node)
        self.tree.append(self.init_node)
        if not res.success:
            return None
        
        initial_guess = self.manhattan_round(res.x, self.integer)
        self.queue.append(initial_guess)
        visited = set()
        visited.add(tuple(initial_guess))
        iter = 1
        best_value = float('inf')

        while len(self.queue) > 0 and  iter <= max_iter:
            iter += 1

            x_fixed = self.queue.popleft()
            new_bounds = self.init_node["bounds"].copy()
            for i, is_int in enumerate(self.integer):
                if is_int:
                    new_bounds[i] = (x_fixed[i], x_fixed[i])
            node = copy(self.init_node)
            node["bounds"] = new_bounds
            res = self.optimize_node(node)
            if verbose:
                print(res)
                print(new_bounds)
                print(x_fixed)


            for neighbor in self.generate_neighbors(x_fixed, self.integer,self.init_node["bounds"]):
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in visited:
                    self.queue.append(neighbor)
                    visited.add(neighbor_tuple)   
            
            if not res.success:
                print("infeasible")
                continue

            if res.fun < best_value:
                self.sol = res
                best_value = res.fun
        if verbose:
            print(f"Number of nodes explored: {iter}")
        return self.sol
        



        
        
# values = np.array([10, 40, 30, 50])   # Item values
# weights = np.array([5, 4, 6, 3])      # Item weights
# capacity = 10                       # Knapsack capacity

# n = len(values)

# # Define the linear programming matrices
# c = -values   # Maximize -> minimize negative value
# A = [weights]  # Single inequality constraint for total weight
# b = [capacity] # Knapsack capacity
# bounds = [(0, 1) for _ in range(n)]  # Relaxed 0-1 constraint
# integer  = [1,1,1,1]

# node = {
#     "c" : c,
#     "A_ub" : A,
#     "b_ub" : b,
#     "A_eq" : None,
#     "b_eq" :  None,
#     "bounds" : bounds,
#     "integer" : integer,
#     "parent" : None,
#     "children" : [],
#     "sol" : None
# }
# init_node = Node(c,A_ub=A,b_ub=b,bounds=bounds,integer=integer)
# solver = BruteForceMILP(node)
# solver.solve(verbose = True, max_iter = 100)
# sol = solver.sol
# print(sol)