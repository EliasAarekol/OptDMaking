# from models.model import Model
from model import Model
import numpy as np

from bnb import BranchAndBound
from brute import BruteForceMILP
class Knapsack(Model):
    def __init__(
            self,
            c,
            w,
            a,
            b,
            W_max,
            ):
        self.c = c # 1 x n_x
        self.w = w # 1 x n_x
        self.a = a # n_V x n_x
        self.b = b # 1 x 1
        self.W_max = W_max
        self.B_t = None

        self.n_value_pieces = a.shape[0]
        self.n_desc_vars = self.w.shape[0]


    def update_state(self,B_t):
        self.B_t = B_t
        return 
    
    def get_LP_formulation(self):
        neg_ones = -1 * np.ones((self.n_value_pieces,1))
        upper = np.hstack((self.a,neg_ones))
        eye = np.eye(self.n_desc_vars)
        middle = np.hstack((eye,np.zeros((self.n_desc_vars,1))))
        lower = np.hstack((self.w,0))
        A = np.vstack((upper,middle,lower))
        
        upper = -self.a @ self.B_t + self.b
        middle = 1 - self.B_t
        lower = self.W_max - self.w @ B_t
        b = np.hstack((upper,middle,lower))
        integer = np.ones((self.n_desc_vars,1))
        integer = np.vstack((integer,0))
        bounds =  [(0, 1) for _ in range(self.n_desc_vars)]
        bounds.append((None,None))

        c = np.hstack((self.c,1))
        A_eq = np.hstack((np.ones(self.n_desc_vars),0))
        A_eq = np.atleast_2d(A_eq).T
        b_eq = 1
        node = {
            "c" : c,
            "A_ub" : A,
            "b_ub" : b,
            "A_eq" : A_eq.T,
            "b_eq" :  b_eq,
            "bounds" : bounds,
            "integer" : integer,
            "parent" : None,
            "children" : [],
            "sol" : None
        }
        return node
    
    # Highly unlikely that this works properly
    def lagrange_gradient(self,x_t,ineq_duals):
        # This doesnt work with experience replay
        w_lambda = ineq_duals[-1]
        dLdw = w_lambda *(self.B_t + x_t)
        dLda = []
        dLdb = []
        for i in range(self.n_value_pieces):
            dLda.append(-ineq_duals[i]*(self.B_t + x_t))
            dLdb.append(-ineq_duals[i])
        dLda = np.array(dLda)
        dLdb = np.array(dLdb)
        return dLdw,dLda,dLdb


    def get_params(self):
        return self.w
        

c =- np.array([1,2,3,5,1])
w = np.array([2,3,1,4,1])
a = np.array([
    [0.1,0.2,0.2,0.1,0.5],
    [0.3,0.4,0.1,0.3,0.2]
    ])
b = np.array([1,2])
W_max = 10
print(w.shape)
B_t = np.array([0,0,0,1,0])
m = Knapsack(c,w,a,b,W_max)
m.update_state(B_t)
node = m.get_LP_formulation()

# print(node)
solver = BranchAndBound(node,None)
sol = solver.solve(verbose = True)
print(sol.x)
x_t = sol.x[0:-1]
print(x_t)
print(sol.ineqlin.marginals)
# dont need eqlin here but maybe good for interface superclass thingy
print(sol.eqlin.marginals)
duals = sol.ineqlin.marginals
dw,da,db = m.lagrange_gradient(x_t,duals)
print("dw",dw)
print("da",da)
print("db",db)

solver = BruteForceMILP(node)

sol = solver.solve(verbose = True)

print(sol.x)
x_t = sol.x[0:-1]
print(x_t)
print(sol.ineqlin.marginals)
# dont need eqlin here but maybe good for interface superclass thingy
print(sol.eqlin.marginals)
duals = sol.ineqlin.marginals
dw,da,db = m.lagrange_gradient(x_t,duals)
print("dw",dw)
print("da",da)
print("db",db)
