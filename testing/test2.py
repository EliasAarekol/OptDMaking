from math import inf
from multiprocessing import Pool

# import time
import cProfile
# from scipy.optimize import linprog
# import scipy


# def optimize_node2(c,A_ub,b_ub,bounds):
#     # return linprog(
#     #     c,
#     #     A_ub,
#     #     b_ub,
#     #     None,
#     #     None,
#     #     bounds
#     #     )
#     return

def foo(a):
    return a
# def optimize_node_para(node,results,index):
#     res =  linprog(
#         node["c"],
#         node["A_ub"],
#         node["b_ub"],
#         node["A_eq"],
#         node["b_eq"],
#         node["bounds"],
#         )
#     results[index] = res

# def manhattan_round(x, int_indices):
#     """Find the closest integer solution by rounding the integer-constrained variables."""
#     x_int = x.copy()
#     for i, is_int in enumerate(int_indices):
#         if is_int:
#             x_int[i] = round(x[i])
#     return x_int

# def generate_neighbors(x, is_integer,bounds):
#     # Needs to handle that the original bounds shouldnt be breached
#     """Generate neighboring integer solutions by varying each integer-constrained variable by ±1."""
#     neighbors = []
#     for i, is_int in enumerate(is_integer):
#         if is_int:
#             l_bound,r_bound = bounds[i]
#             l_bound = l_bound if l_bound is not None else -inf
#             r_bound = r_bound if r_bound is not None else inf
#             x_new = x.copy()
#             x_new[i] = min(x[i] + 1,r_bound)
#             neighbors.append(x_new)
#             x_new = x.copy()
#             x_new[i] = max(x[i] - 1,l_bound)
#             neighbors.append(x_new)
#     return neighbors
# def worker_function(c_arr, A_ub_arr, b_ub_arr, bounds):
#     c_local = np.frombuffer(c_arr, dtype=np.float64)  # Convert back to NumPy
#     A_ub_local = np.frombuffer(A_ub_arr, dtype=np.float64)
#     b_ub_local = np.frombuffer(b_ub_arr, dtype=np.float64)
    
#     return optimize_node2(c_local, A_ub_local, b_ub_local, bounds)
def bruteForceSolveMILP(node,max_iter=10000, store_pool=False, verbose=False):
    # integer = node["integer"]
    # pool = []
    # queue = deque()
    # res = optimize_node(node)
    # pool.append(node)
    # if not res.success:
    #     return None

    # initial_guess = manhattan_round(res.x, integer)
    # queue.append(initial_guess)
    # visited = set()
    # visited.add(tuple(initial_guess))
    # visited = set()
    # visited.add(tuple(initial_guess))
    # iter = 1
    # best_value = float('inf')
    # orig_node = node
    # orig_bounds = node["bounds"].copy()
    # while len(queue) > 0 and  iter <= max_iter:
    #     iter += 1

    #     x_fixed = queue.popleft()
    #     new_bounds = orig_bounds.copy()
    #     for i, is_int in enumerate(integer):
    #         if is_int:
    #             new_bounds[i] = (x_fixed[i], x_fixed[i])
    #     node = copy(orig_node)
    #     node["bounds"] = new_bounds
    #     pool.append(node)
    #     for neighbor in generate_neighbors(x_fixed, integer,orig_node["bounds"]):
    #         neighbor_tuple = tuple(neighbor)
    #         if neighbor_tuple not in visited:
    #             queue.append(neighbor)
    #             visited.add(neighbor_tuple)
    # # with Pool(8) as p:
    # #     res = p.map(optimize_node,pool)
    # # print(res)
    # jobs = []
    # results = [None for p in pool]
    # index = 0
    # for node,res in zip(pool,results):
    #     # print(node)
    #     process = Process(target = optimize_node_para,args = (node,results,index))
    #     jobs.append(process)
    # #     index += 1
    # pool = [deepcopy(prob) for prob in pool]


    # c = orig_node["c"]
    # c_array = Array("d", len(c.flatten()), lock=False)
    # c_array[:] = c.flatten()[:]

    # A_ub = orig_node["A_ub"]
    # A_ub_array = Array("d", len(A_ub.flatten()), lock=False)
    # A_ub_array[:] = A_ub.flatten()[:]

    # b_ub = orig_node["b_ub"]
    # b_ub_array = Array("d", len(b_ub.flatten()), lock=False)
    # b_ub_array[:] = b_ub.flatten()[:]

    # A_eq = orig_node["A_eq"]
    # A_eq_array = None if A_eq is None else Array("d", len(A_eq.flatten()), lock=False)
    # if A_eq_array is not None:
    #     A_eq_array[:] = A_eq.flatten()[:]

    # b_eq = orig_node["b_eq"]
    # b_eq_array = None if b_eq is None else Array("d", len(b_eq.flatten()), lock=False)
    # if b_eq_array is not None:
    #     b_eq_array[:] = b_eq.flatten()[:]

    # c = orig_node["c"].flatten()
    # c_array = Array("d", c, lock=False)  # No lock for better performance

    # A_ub = orig_node["A_ub"].flatten()
    # A_ub_array = Array("d", A_ub, lock=False)

    # b_ub = orig_node["b_ub"].flatten()
    # b_ub_array = Array("d", b_ub, lock=False)

    # bounds = orig_node["bounds"]
    # bounds_array = Array("d", len(bounds.flatten()), lock=False)
    # bounds_array[:] = bounds.flatten()[:]
    # c_list = orig_node["c"]
    # A_ub_list = orig_node["A_ub"]
    # b_ub_list = orig_node["b_ub"]
    # bounds_list = list(orig_node["bounds"])

    # inputs = [(deepcopy(c_list), deepcopy(A_ub_list), deepcopy(b_ub_list), deepcopy(node["bounds"])) for node in pool]
    inputs = [0 for n in range(1000)]
    with Pool() as p:
        p.map(foo,inputs)
    # p = Pool()
    # results = p.starmap(optimize_node2,inputs)
    # p.close()
    # p.join()
    # results = []
    # for input in inputs:
    #     results.append(p.apply_async(optimize_node2,args = input))
    # for res in results:
    #     res.get()
    # p.close()
    # p.join()
        
    # with Pool() as p:
    #     results = p.starmap(optimize_node2, inputs)
    #         #     break
        
    # for j in jobs:
    #     j.start()

    # # Ensure all of the processes have finished
    # for j in jobs:
    #     j.join()
    # return results

def main():
                
    # values = np.array([1,2,2,5,1,1,1,1,1,1,1])
    # weights = np.array([2,3,1,4,1,1,1,1,1,1,1])     # Item weights
    # capacity = 10                       # Knapsack capacity

    # n = len(values)

    # # Define the linear programming matrices
    # c = -values   # Maximize -> minimize negative value
    # A = np.array([weights])  # Single inequality constraint for total weight
    # b = np.array([capacity]) # Knapsack capacity
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
    node = None
    # # init_node = Node(c,A_ub=A,b_ub=b,bounds=bounds,integer=integer)
    # start = time.time()
    sol = bruteForceSolveMILP(node)
    # print(sol)
    # print(time.time()-start)
    # print(len(sol))
    # start = time.time()
    # solver = BruteForceMILP(node)
    # _,sol = solver.solve(store_pool = True ,verbose = False, max_iter = 100)
    # print(time.time()-start)
    # print(len(sol))
    
    # sol = solver.sol
    # print(sol)

if __name__ == "__main__":
    cProfile.run("main()",sort = "time")
    # main()
