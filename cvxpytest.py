import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer
from scipy.optimize import linprog
import numpy as np
from models import knapsack, policy
c = np.array([-1, 4])  # Objective function: minimize -x1 + 4x2

# Inequality constraints: -3x1 + x2 <= 6 and x1 + 2x2 <= 4
A_ub = np.array([[-3, 1], [1, 2]])
b_ub = np.array([6, 3])

# Equality constraint: x1 + x2 = 2 (optional)
A_eq = np.array([[1, 1]])
b_eq = np.array([2])

# Variable bounds: x1 >= 0, x2 >= 0
bounds = [(0, None), (0, None)]

sol = linprog( 
    -c,
    A_ub,
    b_ub,
    A_eq,
    b_eq,
    bounds,
    method="highs-ipm"
    )
print(sol)
print(A_ub@sol.x)

c_b = torch.tensor([-1, 4], dtype=torch.float32)  # Objective function

# Inequality constraints: -3x1 + x2 <= 6 and x1 + 2x2 <= 4
A_ub_b = torch.tensor([[-3, 1], [1, 2]], requires_grad=True, dtype=torch.float32)
b_ub_b = torch.tensor([6, 4], requires_grad=True, dtype=torch.float32)

# Equality constraint: x1 + x2 = 2
A_eq_b = torch.tensor([[1, 1]], requires_grad=True, dtype=torch.float32)
b_eq_b = torch.tensor([2], requires_grad=True, dtype=torch.float32)

# Define optimization variables
x = cp.Variable(c_b.shape[0])
c = cp.Parameter(c_b.shape[0])  # ✅ Change to Parameter
A_ub = cp.Parameter(A_ub_b.shape)
b_ub = cp.Parameter(b_ub_b.shape[0])
A_eq = cp.Parameter(A_eq_b.shape)
b_eq = cp.Parameter(b_eq_b.shape[0])

# Constraints
constraints = [x >= 0, A_ub @ x <= b_ub, A_eq @ x == b_eq]
objective = cp.Minimize(-c @ x)  # ✅ Uses c as a parameter
problem = cp.Problem(objective, constraints)

assert problem.is_dpp()  # ✅ Now this should pass

cvxpylayer = CvxpyLayer(problem, parameters=[c, A_eq, b_eq, A_ub, b_ub], variables=[x])

# Solve the problem
solution, = cvxpylayer(c_b, A_eq_b, b_eq_b, A_ub_b, b_ub_b,solver_args={"solve_method":"ECOS"})

print(solution)
objective_value = (-c_b @ solution)
print(objective_value)

# Compute gradients
objective_value.backward()
print(b_ub_b.grad)

constraint_slack = A_ub_b @ solution - b_ub_b
print("Constraint Slack:", constraint_slack)


c = np.array([1,2,2,5,1])
w_true = np.array([2,3,1,4,1])
# w= np.array([2,3,1,4,1])
np.random.seed(0)
# w = w_true + np.random.uniform(-0.5,0.5,size=(5,))
# print("w:",w)
# a = np.array([
#     [0.1,0.2,0.2,0.1,0.5],
#     [0.3,0.4,0.1,0.3,0.2]
# ])
# b = np.array([1,2])
a = np.array([
    [0.,0.,0.,0.,0.],
    [0.,0.,0.,0.,0.]
])
b = np.array([0,0])
W_max = [10]
m = knapsack.Knapsack(-c,w_true,a,b,W_max)



init_state = np.array([0,0,0,0,0])
m.update_state(init_state)
node = m.get_LP_formulation()

sol = linprog( 
    node["c"],
   node["A_ub"],
   node["b_ub"],
    node["A_eq"],
    node["b_eq"],
    node["bounds"],
    method="highs-ipm"
    )
print(sol)

w_true = torch.tensor(w_true,requires_grad= True,dtype=torch.float64)
node["c"] = torch.tensor(node["c"],dtype = torch.float64)



x = cp.Variable(node["c"].shape[0]-1)
c = cp.Parameter(node["c"].shape[0]-1)
w = cp.Parameter(w_true.shape[0])

constraints = [x[:-1] >= 0, x[:-1] <= 1, w @ (m.B_t + x[:-1] ) == W_max + x[-1],  x + m.B_t <= 1,cp.sum(x[:-1]) <= 1]
objective = cp.Minimize(c @ x)  # ✅ Uses c as a parameter
problem = cp.Problem(objective, constraints)

assert problem.is_dpp()  # ✅ Now this should pass

cvxpylayer = CvxpyLayer(problem, parameters=[c, w], variables=[x])

# Solve the problem
solution, = cvxpylayer(node["c"][:-1],w_true,solver_args={"solve_method":"ECOS"})
print(solution)
objective_value = (node["c"][:-1] @ solution)
print(objective_value)
objective_value.backward()

print("w",w_true.grad)

# ✅ Change to Parameter
x = cp.Variable(node["c"].shape)
c = cp.Parameter(node["c"].shape)

A_ub = cp.Parameter(node["A_ub"].shape)
b_ub = cp.Parameter(node["b_ub"].shape)
A_eq = cp.Parameter(node["A_eq"].shape)
b_eq = cp.Parameter(node["b_eq"].shape)
node["c"] = torch.tensor(node["c"],dtype = torch.float64)
node["A_ub"] = torch.tensor(node["A_ub"],dtype = torch.float64,requires_grad= True)
node["b_ub"] = torch.tensor(node["b_ub"],dtype = torch.float64,requires_grad= True)
node["A_eq"] = torch.tensor(node["A_eq"],dtype = torch.float64,requires_grad= True)
node["b_eq"] = torch.tensor(node["b_eq"],dtype = torch.float64,requires_grad= True)

constraints = [x[:-2] >= 0, x[:-2] <= 1, A_ub @ x <= b_ub, A_eq @ x == b_eq,x[2] == 1]

objective = cp.Minimize(c @ x)  # ✅ Uses c as a parameter
problem = cp.Problem(objective, constraints)


assert problem.is_dpp()  # ✅ Now this should pass

cvxpylayer = CvxpyLayer(problem, parameters=[c, A_ub,b_ub,A_eq,b_eq], variables=[x])


solution, = cvxpylayer(node["c"],node["A_ub"],node["b_ub"],node["A_eq"],node["b_eq"],solver_args={"solve_method":"ECOS"})
objective_value = (node["c"] @ solution)
objective_value.backward()
print("solution",solution)
print("objective_value",objective_value)
print("A_eq",node["b_ub"].grad)

# np.random.seed(0)
# vals_np = np.random.normal(0,1, size = (4,))
# dphidtheta_np = np.random.normal(0,1,size = (4,2))
# print("vals",vals_np)
torch.manual_seed(0)

vals = torch.rand((4,))
vals_np = vals.numpy()
vals.requires_grad = True
dphidtheta = torch.rand((4,2)) #
dphidtheta_np = dphidtheta.numpy()
dphidtheta.requires_grad = True



print(vals)
pol = policy.policy_dist_torch(vals,1) # pi
print(pol)
log_pol = torch.log(pol)
log_pol[2].backward() # dpi/dphi
print("Gradient of policy w.r.t. objective values:",vals.grad)

# jac = torch.autograd.functional.jacobian(policy.policy_dist_torch,vals)
# print(jac)
nab = policy.nabla_log_pi(dphidtheta_np[2],vals_np,dphidtheta_np,beta = 1) # dpi/dtheta
print(nab)

grad_log_pol = vals.grad  # This is d(log π) / dvals
expected_nabla_log_pi = grad_log_pol @ dphidtheta
print(expected_nabla_log_pi)
error = np.linalg.norm(nab - expected_nabla_log_pi.detach().numpy())
print(f"Error between manual and PyTorch gradients: {error}")
print(nab,expected_nabla_log_pi)