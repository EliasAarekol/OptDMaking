import numpy as np
from src.models.gg1queue import GG1Queue
from src.solvers.bnb import BranchAndBound,BranchAndBoundRevamped
from src.solvers.brute import BruteForcePara
from multiprocessing import Pool
import itertools
import random


def gen_feas_action_space(num_actions, n, max_value, node):
    """
    Generate a feasible action space satisfying MILP constraints.

    Args:
        num_actions (int): Number of actions to return.
        n (int): Number of decision variables.
        max_value (int): Maximum value for each variable.
        node (dict): Dictionary with keys:
            - "A_eq" (equality constraint matrix)
            - "b_eq" (equality constraint vector)
            - "A_ub" (inequality constraint matrix)
            - "b_ub" (inequality constraint vector)
            - "bounds" (list of (lower, upper) bounds for each variable)

    Returns:
        np.array: Feasible actions (subset of `num_actions` actions).
    """

    # Generate all possible integer actions in the range [0, max_value]
    all_actions = np.array(list(itertools.product(range(max_value + 1), repeat=n)))

    # Apply equality constraints (A_eq * x = b_eq)
    if node["A_eq"] is not None and node["b_eq"] is not None:
        eq_cond = np.all(np.isclose(node["A_eq"] @ all_actions.T, node["b_eq"][:, None]), axis=0)
    else:
        eq_cond = np.ones(all_actions.shape[0], dtype=bool)  # No equality constraints

    # Apply inequality constraints (A_ub * x ≤ b_ub)
    if node["A_ub"] is not None and node["b_ub"] is not None:
        ineq_cond = np.all(node["A_ub"] @ all_actions.T <= node["b_ub"][:, None], axis=0)
    else:
        ineq_cond = np.ones(all_actions.shape[0], dtype=bool)  # No inequality constraints

    # Apply variable bounds
    bound_low = np.array([bound[0] if bound[0] is not None else -float('inf') for bound in node["bounds"]])
    bound_up = np.array([bound[1] if bound[1] is not None else float('inf') for bound in node["bounds"] ])
    # bound_cond = np.logical_and.reduce((all_actions >= bound_low, all_actions <= bound_up))
    bound_cond = np.all((all_actions >= bound_low) & (all_actions <= bound_up), axis=1)

    # Find feasible actions (satisfy all constraints)
    feasible_mask = np.logical_and.reduce((eq_cond, ineq_cond, bound_cond))
    feasible_actions = all_actions[feasible_mask]

    # If fewer feasible actions exist than requested, return all
    if len(feasible_actions) <= num_actions:
        return feasible_actions

    # Otherwise, randomly sample num_actions from feasible set
    selected_indices = np.random.choice(len(feasible_actions), num_actions, replace=False)
    return feasible_actions[selected_indices]

def assign_obj_bounds(actions,bounding_sets):
    action_bounds = {}

    for action in actions:
        best_ub = float('inf')  # Start with a very high upper bound
        for bound_set in bounding_sets:
            conditions = bound_set["conditions"]
            ub = bound_set["upper_bound"]

            # Check if action satisfies all conditions in this bounding set
            # l = [var,op,val for var,op,val in conditions ]
            satisfies_all = all(
                eval(f"{action[var]} {op} {val}") for var, op, val in conditions if var < len(action)
            )

            # If conditions are met, update best upper bound
            if satisfies_all:
                best_ub = min(best_ub, ub)

        # Assign the best found upper bound (or None if no bound applies)
        # action_bounds[tuple(action.items())] = best_ub if best_ub != float('inf') else None

    return best_ub


if __name__ == "__main__":
    prob_size = 5
    np.random.seed(0)
    c = -np.random.randint(0,10,size=(prob_size,))
    # a = np.random.uniform(0,1,size = (2,2))
    # b = np.random.uniform(0,1,size=(2,))
    a = np.zeros((2,prob_size))
    b = np.zeros((2,))
    wx = np.random.uniform(1,2,size = (prob_size,))
    W_x_max = np.random.randint(20,30,size = (1,) )
    bounds = [(0,None) for _ in range(len(c))]
    integer = np.ones_like(c)
    
    model = GG1Queue(c,a,b,wx,W_x_max,bounds,integer)
    
    It_prev = np.random.randint(0,5,size=(prob_size,))
    Dt = np.random.randint(0,5,size=(prob_size,))
    
    
    model.update_state(Dt,It_prev)
    
    node = model.get_LP_formulation()
    
    print(node)
    solver = BruteForcePara(4)
    sols = solver.solve(node)
    solver = BranchAndBoundRevamped()
    sol2 = solver.solve(node,verbose = True)
    print("Dt",Dt)
    print("It_prev",It_prev)
    print(sol2)
    # print(sol)
    print(W_x_max)
    faths = [sol for sol in sol2 if sol["fathomed"]]
    not_faths = [sol for sol in sol2 if not sol["fathomed"]]
    print(faths[0]["conds"])
    print(faths[1]["conds"])
    print(faths[2]["conds"])
    print(faths[3]["conds"])
    print(not_faths[0]["conds"])
    
    bounding_set = [{"conditions" : sol["conds"] , "upper_bound" : sol["fun"]} for sol in sol2 if sol["fathomed"]]
    # print(bounding_set)
    
    node = {
            "c" : node["c"][:prob_size],
            "A_ub" : node["A_ub"][:,:prob_size],
            "b_ub" : node["b_ub"],
            "A_eq" : None,
            "b_eq" :  node["b_eq"],
            "bounds" : node["bounds"][:prob_size],
    }
    
    a_space = gen_feas_action_space(float('inf'),5,10,node)
    print(len(a_space))
    action_bounds = assign_obj_bounds(a_space,bounding_set)
    print(action_bounds)
    # sol_best = min(sol,key = lambda x : x["fun"])
    # print(sol_best)
    # sol_best = min(solver.pool,key = lambda x : x.fun)
    # print(sol_best)

    # print(sol)
    # print(len(sol))
    # print(len(solver.pool))
    # print(wx)
    
    
    # print(node)