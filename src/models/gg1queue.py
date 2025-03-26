
from src.models.model_template import Model
import numpy as np
from copy import copy

class GG1Queue(Model): # Fix D
    def __init__(self,c,a,b,w_x,W_x_max,bounds,integer):
        self.c = c
        self.a = a
        self.b = b
        self.w_x = w_x
        self.W_x_max = W_x_max
        self.bounds = bounds
        self.integer = integer
        self.Dt = None
        self.It_prev = None
        
        
    def update_state(self,Dt,It_prev):
        self.Dt = Dt
        self.It_prev = It_prev
        
    def get_LP_formulation(self):
        state_size = self.It_prev.shape
        c = np.hstack((self.c,np.zeros(state_size),1)) # Append nxt state estimate and value func estimate
        
        xt_zeros = np.zeros_like(self.a)
        neg_ones = -1 * np.ones((self.a.shape[0],1))
        A_ub_upper = np.hstack((xt_zeros,self.a,neg_ones))
        b_ub_upper = -self.b
        x_eye = np.eye(state_size[0])
        I_eye = np.eye(state_size[0])
        zero_vec = np.zeros((state_size[0],1))
        A_eq_upper = np.hstack((x_eye,-I_eye,zero_vec))
        b_eq_upper = self.Dt - self.It_prev
        
        
        
        A_eq_lower = np.atleast_2d(np.hstack((self.w_x.T,xt_zeros[0,:],0))) 
        
        b_eq_lower = np.array(self.W_x_max)
        
        A_ub = np.atleast_2d(np.vstack((A_ub_upper,A_eq_lower)))
        b_ub = np.hstack((b_ub_upper,b_eq_lower))

        # A_eq = np.vstack((A_eq_upper,A_eq_lower))
        # b_eq = np.hstack((b_eq_upper,b_eq_lower))
        
        bounds = copy(self.bounds)
        extra_bounds = [(0,None) for _ in range(state_size[0])]
        extra_bounds.append((None,None))
        bounds.extend(extra_bounds)
        integer = np.hstack((self.integer,self.integer,0))
        
        node = {
            "c" : c,
            "A_ub" : A_ub,
            "b_ub" : b_ub,
            "A_eq" : A_eq_upper,
            "b_eq" :  b_eq_upper,
            "bounds" : bounds,
            "integer" : integer,
            "parent" : None,
            "children" : [],
            "sol" : None
        }
        return node 
        
    def lagrange_gradient(self):
        return super().lagrange_gradient()
    def get_params(self):
        return super().get_params()
    def update_params(self, grad, lr):
        return super().update_params(grad, lr)
    
    
    
