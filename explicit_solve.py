# Example construction in NumPy
# Assuming c is a NumPy array of shape (n,)
# Assuming n, m, T are defined
import numpy as np
from scipy.sparse import lil_matrix, hstack, vstack, identity, kron


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
T = 7

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

sols = solver.solve(node)
best = sols[0]
for sol in sols:
   if sol["fun"] < best["fun"]:
      best = sol

print(best["x"],best["fun"])

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