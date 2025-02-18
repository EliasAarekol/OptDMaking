from scipy.optimize import linprog
from collections import deque
import numpy as np
from copy import deepcopy
from math import floor,ceil



# Node dictionary template

example_node = {
    "c" : None,
    "A_ub" : None,
    "b_ub" : None,
    "A_eq" : None,
    "b_eq" : None,
    "bounds" : None,
    "integer" : None,
    "parent" : None,
    "children" : [],
    "sol" : None
}






class BranchAndBound:
    """
        Implements a basic branch and bound algorithm, takes a dictionary defining the inital problem node as its input.
    
    """
    def __init__(self,init_node,sense):
        self.init_node = init_node
        self.integer = self.init_node["integer"]
        self.tree = []
        self.queue = deque()
        self.sol = None
        self.end_node = None


    def optimize_node(self,node):
        return linprog(
            node["c"],
            node["A_ub"],
            node["b_ub"],
            node["A_eq"],
            node["b_eq"],
            node["bounds"],
            )

# Handle infeasible sub sol
# Add parent and child relation

    def branch(self,node,res):
        branch_var = np.argmax(np.abs(res.x) - np.floor(np.abs(res.x)))

        left_branch = deepcopy(node)
        right_branch = deepcopy(node)

        left_branch["bounds"][branch_var] = (left_branch["bounds"][branch_var][0],floor(res.x[branch_var]))
        right_branch["bounds"][branch_var] = (ceil(res.x[branch_var]),left_branch["bounds"][branch_var][1] )

        left_branch["parent"] = node
        right_branch["parent"] = node
        return left_branch,right_branch

        
    def all_integer(self,x):
       return all( [not(var.is_integer() ^ bool(i)) for var,i in zip(x,self.integer)])


    def solve(self):
        res = self.optimize_node(self.init_node)
        if not res.success:
            return None
        self.tree.append(self.init_node)
        if self.all_integer(res.x):
            print(res.x[0].is_integer())
            self.sol = res
            self.init_node["sol"] = res
            self.end_node = self.init_node
            return self.sol
        
        sol_rounded = np.floor(res.x) # Does this work for negative solutions?

        ub = self.init_node["c"] @ sol_rounded
        l,r = self.branch(self.init_node,res)
        self.queue.append(l)
        self.queue.append(r)
        self.init_node["children"].append(l)
        self.init_node["children"].append(r)


        while len(self.queue) > 0:
            node = self.queue.popleft()
            self.tree.append(node)
            res = self.optimize_node(node)
            node["sol"] = res

            if not res.success:
                print("infeasible")
                continue

            if self.all_integer(res.x):
                if res.fun <= ub:
                    ub = res.fun
                    self.sol = res
                    self.end_node = node

            elif res.fun <= ub:
                l,r = self.branch(node,res)

                self.queue.append(l)
                self.queue.append(r)
                node["children"].append(l)
                node["children"].append(r)
        return self.sol


