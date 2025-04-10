# Example construction in NumPy
# Assuming c is a NumPy array of shape (n,)
# Assuming n, m, T are defined
import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack, identity, kron,csr_matrix


num_cons = 6
prob_size = 3
num_pieces = 2
np.random.seed(5)
aA = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
aB = np.random.uniform(0,0.1,size = (num_pieces,prob_size))
b = np.random.uniform(0,.1,size=(num_pieces,))
c = -np.random.randint(0,10,size=(prob_size,))
state = np.random.randint(2,size = prob_size)
A = np.random.randint(0,2,size = (prob_size,prob_size))
B = np.random.randint(0,2,size = (prob_size,prob_size))
D = np.random.uniform(0,1,size = (num_cons,prob_size))
E = np.random.uniform(0,2,size = (num_cons,prob_size))
F = np.random.uniform(5,10,size = (num_cons))
# F = np.random.uniform(1,3,size = (num_cons))
# c = - np.random.randint(0,10,size=(prob_size,))
T = 5

m = prob_size
n = prob_size


c_agg_x = np.tile(c, T)  # Repeats c T times
c_agg_s = np.zeros((T + 1) * m)
c_agg = np.hstack([c_agg_x, c_agg_s])


# Assuming A, B are NumPy arrays or sparse matrices of appropriate shape
# n = dim(x_t), m = dim(s_t), T = horizon

N = T * n + (T + 1) * m
num_eq_constraints = T * m
A_eq = lil_matrix((num_eq_constraints, N))
I_m = identity(m, format='csr') # Use sparse identity

for t in range(T):
    row_start = t * m
    row_end = (t + 1) * m

    # Column indices for x_t
    col_start_xt = t * n
    col_end_xt = (t + 1) * n

    # Column indices for s_t
    col_start_st = T * n + t * m
    col_end_st = T * n + (t + 1) * m

    # Column indices for s_{t+1}
    col_start_st1 = T * n + (t + 1) * m
    col_end_st1 = T * n + (t + 2) * m

    A_eq[row_start:row_end, col_start_xt:col_end_xt] = B
    A_eq[row_start:row_end, col_start_st:col_end_st] = A
    A_eq[row_start:row_end, col_start_st1:col_end_st1] = -I_m

# Convert to a more efficient format like CSR or CSC for calculations
A_eq = A_eq.tocsr()

# Right-hand side is all zeros
b_eq = np.zeros(num_eq_constraints)


from scipy.sparse import block_diag

# Assuming D, E, F are NumPy arrays or sparse matrices
k = F.shape[0] # = number of inequality constraints per step
num_ineq_constraints = T * k

# Build the block for x variables
A_ub_x = block_diag([E] * T, format='csr') # Size (T*k) x (T*n)

# Build the block for s variables (s_0 to s_{T-1})
A_ub_s_main = block_diag([D] * T, format='csr') # Size (T*k) x (T*m)

# Create the zero block for s_T variable columns
A_ub_s_T_zeros = lil_matrix((num_ineq_constraints, m)) # Size (T*k) x m

# Combine the s blocks
A_ub_s = hstack([A_ub_s_main, A_ub_s_T_zeros], format='csr') # Size (T*k) x ((T+1)*m)

# Combine x and s blocks
A_ub = hstack([A_ub_x, A_ub_s], format='csr') # Size (T*k) x N

# Right-hand side: stack F T times
# Ensure F is a 1D array
if F.ndim > 1:
   F_flat = F.flatten()
else:
   F_flat = F
b_ub = np.tile(F_flat, T) # Repeats F T times

bounds = [(0,9) for _ in range(A_ub.shape[1])]
integer = [1 for _ in range(A_ub.shape[1])]

from src.solvers.bnb import BranchAndBoundRevamped

solver = BranchAndBoundRevamped()

node = {
    "c" : c_agg,
    "A_ub" : A_ub,
    "b_ub" : b_ub,
    "A_eq" : A_eq,
    "b_eq" : b_eq,
    "bounds" : bounds,
    "integer" : integer,
}

# sols = solver.solve(node)
# best = sols[0]
# for sol in sols:
#    if sol["fun"] < best["fun"]:
#       best = sol

# print(best["x"],best["fun"])

# actions = best["x"][:T*n]
# print(actions)

print( B@ np.array([1,0,1]))
print(A@np.array([1,0,1]))
# prod = B @ np.array([0,1,0])



from src.gym_envs.arb_discrete_gym_env import Arb_binary

gym_model = Arb_binary(-c,np.zeros_like(c),A,B,D,E,F,1)

gym_model.state = np.array([0,0,0])
obs,reward,terminated,_,info = gym_model.step(np.array([0,1,0]))

print(obs,reward,terminated)
obs,reward,terminated,_,info = gym_model.step(np.array([0,0,0]))
print(obs,reward,terminated)
obs,reward,terminated,_,info = gym_model.step(np.array([1,0,0]))
print(obs,reward,terminated)
print(A)
print(B)
print(D)
print(E)
print(F)
print(c)
print(list(range(2)))





def formulate_lp_with_initial_state(c, A, B, D, E, F, T, s_initial):
    """
    Formulates the time-dependent LP into standard form min c'z s.t.
    A_eq z = b_eq, A_ub z <= b_ub, assuming a fixed initial state s_0.

    Args:
        c (np.ndarray): Cost vector for x_t (dim n).
        A (np.ndarray or sparse matrix): State transition matrix for s_t (dim m x m).
        B (np.ndarray or sparse matrix): State transition matrix for x_t (dim m x n).
        D (np.ndarray or sparse matrix): Inequality matrix for s_t (dim k x m).
        E (np.ndarray or sparse matrix): Inequality matrix for x_t (dim k x n).
        F (np.ndarray): Right-hand side for inequality constraints (dim k).
        T (int): Time horizon (number of steps, x_t goes from 0 to T-1).
        s_initial (np.ndarray): The fixed initial state vector s_0 (dim m).

    Returns:
        tuple: (c_agg, A_eq, b_eq, A_ub, b_ub)
               Ready for scipy.optimize.linprog (bounds need to be added separately).
               Matrices A_eq and A_ub are returned as CSR sparse matrices.
    """
    # Ensure inputs are numpy arrays for shape info
    c = np.asarray(c)
    s_initial = np.asarray(s_initial)
    F = np.asarray(F)

    # --- Dimensions ---
    n = B.shape[1]  # Dimension of x_t
    m = A.shape[0]  # Dimension of s_t
    if D is not None and E is not None:
       k = D.shape[0] # Number of inequality constraints per step
    else: # Handle case with no inequality constraints D, E, F
        k = 0


    if s_initial.shape[0] != m:
        raise ValueError(f"s_initial dimension ({s_initial.shape[0]}) must match A rows ({m})")
    if c.shape[0] != n:
        raise ValueError(f"c dimension ({c.shape[0]}) must match B columns ({n})")
    if k > 0 and F.shape[0] != k:
         raise ValueError(f"F dimension ({F.shape[0]}) must match D rows ({k})")


    N = T * n + (T + 1) * m # Total number of variables in z

    # --- Aggregated Cost Vector c_agg ---
    c_agg_x = np.tile(c, T)
    c_agg_s = np.zeros((T + 1) * m)
    c_agg = np.hstack([c_agg_x, c_agg_s])

    # --- Equality Constraints (Dynamics) A_eq_dynamics z = 0 ---
    num_eq_dynamics = T * m
    A_eq_dynamics = lil_matrix((num_eq_dynamics, N))
    I_m = identity(m, format='csr') # Use sparse identity

    for t in range(T):
        row_start = t * m
        row_end = (t + 1) * m

        col_start_xt = t * n
        col_end_xt = (t + 1) * n

        col_start_st = T * n + t * m
        col_end_st = T * n + (t + 1) * m

        col_start_st1 = T * n + (t + 1) * m
        col_end_st1 = T * n + (t + 2) * m

        A_eq_dynamics[row_start:row_end, col_start_xt:col_end_xt] = B
        A_eq_dynamics[row_start:row_end, col_start_st:col_end_st] = A
        A_eq_dynamics[row_start:row_end, col_start_st1:col_end_st1] = -I_m

    b_eq_dynamics = np.zeros(num_eq_dynamics)

    # --- Equality Constraints (Initial State) A_eq_s0 z = s_initial ---
    num_eq_s0 = m
    A_eq_s0 = lil_matrix((num_eq_s0, N))
    s0_col_start = T * n # Column index where s_0 variables begin
    s0_col_end = T * n + m
    A_eq_s0[:, s0_col_start:s0_col_end] = I_m

    b_eq_s0 = s_initial # RHS is the fixed initial state

    # --- Combine Equality Constraints ---
    A_eq = vstack([A_eq_dynamics, A_eq_s0], format='csr')
    b_eq = np.concatenate([b_eq_dynamics, b_eq_s0])

    # --- Inequality Constraints A_ub z <= b_ub ---
    if k > 0:
        num_ineq = T * k
        A_ub_x = block_diag([E] * T, format='csr') # Size (T*k) x (T*n)
        A_ub_s_main = block_diag([D] * T, format='csr') # Size (T*k) x (T*m) (for s_0 to s_{T-1})
        A_ub_s_T_zeros = csr_matrix((num_ineq, m)) # Zero block for s_T columns, Size (T*k) x m
        A_ub_s = hstack([A_ub_s_main, A_ub_s_T_zeros], format='csr') # Size (T*k) x ((T+1)*m)
        A_ub = hstack([A_ub_x, A_ub_s], format='csr') # Size (T*k) x N

        if F.ndim > 1:
            F_flat = F.flatten()
        else:
            F_flat = F
        b_ub = np.tile(F_flat, T)
    else: # No inequality constraints
        # Create empty structures as placeholders or handle as needed by solver
        A_ub = None # Or csr_matrix((0, N)) depending on solver needs
        b_ub = None # Or np.array([])

    return c_agg, A_eq, b_eq, A_ub, b_ub

# from time import time
# c_agg_2,A_eq_2,b_eq_2,A_ub_2,b_ub_2 = formulate_lp_with_initial_state(c,A,B,D,E,F,5,np.array([0,0,1]))


# bounds = [(0,9) for _ in range(A_ub_2.shape[1])]
# integer = [1 for _ in range(A_ub_2.shape[1])]

# start = time()

# node = {
#     "c" : c_agg_2,
#     "A_ub" : A_ub_2,
#     "b_ub" : b_ub_2,
#     "A_eq" : A_eq_2,
#     "b_eq" : b_eq_2,
#     "bounds" : bounds,
#     "integer" : integer,
# }


# sols = solver.solve(node)

# print(f"Took {time()-start} seconds" )
# # print(c_agg == c_agg_2)
# # print(A_eq == A_eq_2)

# best = sols[0]
# for sol in sols:
#    if sol["fun"] < best["fun"]:
#       best = sol

# print(best["x"],best["fun"])


import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack, identity, block_diag, csr_matrix, diags

def formulate_milp_with_relaxation(c, A, B, D, E, F, T, s_initial, penalty_factor):
    """
    Formulates the time-dependent problem as a Mixed-Integer Linear Program (MILP)
    in standard form min c'z s.t. A_eq z = b_eq, A_ub z <= b_ub,
    with a fixed initial state s_0, slack variables y_t for inequality constraints,
    integrality constraints (x, s integer; y continuous), and predefined bounds.

    Args:
        c (np.ndarray): Cost vector for x_t (dim n).
        A (np.ndarray or sparse matrix): State transition matrix for s_t (dim m x m).
        B (np.ndarray or sparse matrix): State transition matrix for x_t (dim m x n).
        D (np.ndarray or sparse matrix): Inequality matrix for s_t (dim k x m).
        E (np.ndarray or sparse matrix): Inequality matrix for x_t (dim k x n).
        F (np.ndarray): Right-hand side for inequality constraints (dim k).
        T (int): Time horizon (number of steps, x_t/y_t go from 0 to T-1).
        s_initial (np.ndarray): The fixed initial state vector s_0 (dim m).
        penalty_factor (float): Cost coefficient for the slack variables y_t.

    Returns:
        tuple: (c_agg, A_eq, b_eq, A_ub, b_ub, N_dims, integrality, bounds)
               Ready for scipy.optimize.linprog.
               Matrices A_eq and A_ub are returned as CSR sparse matrices.
               N_dims contains dimensions (N, n_x, n_s, n_y) for help with results.
               integrality is a 1D NumPy array specifying integer (1) or continuous (0) vars.
               bounds is a list of (min, max) tuples for each variable in z.
    """
    # Ensure inputs are numpy arrays for shape info
    c = np.asarray(c)
    s_initial = np.asarray(s_initial)
    F = np.asarray(F)

    # --- Dimensions ---
    n = B.shape[1]  # Dimension of x_t
    m = A.shape[0]  # Dimension of s_t
    if D is not None and E is not None:
        k = D.shape[0] # Number of inequality constraints per step
    else: # Handle case with no inequality constraints D, E, F
        k = 0


    if s_initial.shape[0] != m:
        raise ValueError(f"s_initial dimension ({s_initial.shape[0]}) must match A rows ({m})")
    if c.shape[0] != n:
        raise ValueError(f"c dimension ({c.shape[0]}) must match B columns ({n})")
    if k > 0 and F.shape[0] != k:
        raise ValueError(f"F dimension ({F.shape[0]}) must match D rows ({k})")

    n_x = T * n              # Total number of x variables
    n_s = (T + 1) * m        # Total number of s variables
    n_y = T                  # Total number of y variables (slack)
    N = n_x + n_s + n_y      # Total number of variables in z
    N_dims = {'total': N, 'x': n_x, 's': n_s, 'y': n_y}

    # --- Aggregated Cost Vector c_agg ---
    c_agg_x = np.tile(c, T)
    c_agg_s = np.zeros(n_s)
    c_agg_y = np.full(n_y, penalty_factor)
    c_agg = np.concatenate([c_agg_x, c_agg_s, c_agg_y])

    # --- Equality Constraints (Dynamics) A_eq_dynamics z = 0 ---
    num_eq_dynamics = T * m
    A_eq_dyn_orig = lil_matrix((num_eq_dynamics, n_x + n_s)) # Matrix for x and s parts
    I_m = identity(m, format='csr')

    for t in range(T):
        row_start = t * m; row_end = (t + 1) * m
        col_xt_start = t * n; col_xt_end = (t + 1) * n
        col_st_start = n_x + t * m; col_st_end = n_x + (t + 1) * m
        col_st1_start = n_x + (t + 1) * m; col_st1_end = n_x + (t + 2) * m

        A_eq_dyn_orig[row_start:row_end, col_xt_start:col_xt_end] = B
        A_eq_dyn_orig[row_start:row_end, col_st_start:col_st_end] = A
        A_eq_dyn_orig[row_start:row_end, col_st1_start:col_st1_end] = -I_m

    # Add zero columns for y variables
    A_eq_dyn = hstack([A_eq_dyn_orig, csr_matrix((num_eq_dynamics, n_y))], format='csr')
    b_eq_dynamics = np.zeros(num_eq_dynamics)

    # --- Equality Constraints (Initial State) A_eq_s0 z = s_initial ---
    num_eq_s0 = m
    A_eq_s0_orig = lil_matrix((num_eq_s0, n_x + n_s)) # Matrix for x and s parts
    s0_col_start = n_x # Column index where s_0 variables begin
    s0_col_end = n_x + m
    A_eq_s0_orig[:, s0_col_start:s0_col_end] = I_m

    # Add zero columns for y variables
    A_eq_s0 = hstack([A_eq_s0_orig, csr_matrix((num_eq_s0, n_y))], format='csr')
    b_eq_s0 = s_initial

    # --- Combine Equality Constraints ---
    A_eq = vstack([A_eq_dyn, A_eq_s0], format='csr')
    b_eq = np.concatenate([b_eq_dynamics, b_eq_s0])

    # --- Inequality Constraints A_ub z <= b_ub ---
    # D s_t + E x_t - 1*y_t <= F
    if k > 0:
        num_ineq = T * k
        # Block for x variables
        A_ub_x = block_diag([E] * T, format='csr') # Size (T*k) x n_x

        # Block for s variables (s_0 to s_{T})
        A_ub_s_main = block_diag([D] * T, format='csr') # Size (T*k) x (T*m) (for s_0 to s_{T-1})
        A_ub_s_T_zeros = csr_matrix((num_ineq, m))      # Zero block for s_T columns, Size (T*k) x m
        A_ub_s = hstack([A_ub_s_main, A_ub_s_T_zeros], format='csr') # Size (T*k) x n_s

        # Block for y variables (y_0 to y_{T-1})
        minus_ones_vector = -np.ones((k, 1))
        A_ub_y = block_diag([minus_ones_vector] * T, format='csr') # Size (T*k) x n_y

        # Combine all blocks
        A_ub = hstack([A_ub_x, A_ub_s, A_ub_y], format='csr') # Size (T*k) x N

        # Right-hand side: stack F T times
        if F.ndim > 1:
            F_flat = F.flatten()
        else:
            F_flat = F
        b_ub = np.tile(F_flat, T)
    else: # No inequality constraints
        A_ub = None
        b_ub = None

    # --- Integrality Constraints ---
    integrality_x = np.ones(n_x, dtype=int)  # x variables are integer
    integrality_s = np.ones(n_s, dtype=int)  # s variables are integer
    integrality_y = np.zeros(n_y, dtype=int) # y variables are continuous
    integrality = np.concatenate([integrality_x, integrality_s, integrality_y])

    # --- Bounds Constraints ---
    # Integer variables (x, s) bounds: (0, 9)
    # Continuous variables (y) bounds: (0, inf) -> (0, None)
    bounds_x = [(0, 9)] * n_x
    bounds_s = [(0, 9)] * n_s
    bounds_y = [(0, None)] * n_y
    bounds = bounds_x + bounds_s + bounds_y

    return c_agg, A_eq, b_eq, A_ub, b_ub, N_dims, integrality, bounds


c_agg_2,A_eq_2,b_eq_2,A_ub_2,b_ub_2,_,integer,bounds = formulate_milp_with_relaxation(c,A,B,D,E,F,4,np.array([6,2,1]),0)

from time import time
# bounds = [(0,9) for _ in range(A_ub_2.shape[1])]
# integer = [1 for _ in range(A_ub_2.shape[1])]

start = time()

node = {
    "c" : c_agg_2,
    "A_ub" : A_ub_2,
    "b_ub" : b_ub_2,
    "A_eq" : A_eq_2,
    "b_eq" : b_eq_2,
    "bounds" : bounds,
    "integer" : integer,
}

print(A_ub_2)

sols = solver.solve(node)

print(f"Took {time()-start} seconds" )
# print(c_agg == c_agg_2)
# print(A_eq == A_eq_2)

best = sols[0]
for sol in sols:
   if sol["fun"] < best["fun"]:
      best = sol

print(best["x"],best["fun"])