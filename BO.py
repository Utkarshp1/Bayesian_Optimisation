import torch
from utils import generate_initial_data, optimize_acq_func_and_get_candidates, get_next_query_point
from GaussianProcess import get_and_fit_simple_custom_gp
from botorch.acquisition.analytic import ExpectedImprovement

class BO:
    '''
    '''
    def __init__(self, obj_fn, dtype, acq_func, grad_acq=None, init_examples=5,
        order=0, budget=20):
        
        self.obj_fn = obj_fn
        self.dtype = dtype
        self.init_examples = init_examples
        self.acq_func = acq_func
        self.grad_acq = grad_acq
        self.order = order
        self.budget = budget
   
    def optimize(self):
        # Generate initial data for GP fiiting
        self.X, self.y, self.grads = generate_initial_data(
            self.obj_fn,
            n=self.init_examples,
            dtype=self.dtype
        )
        
        for i in range(self.budget):
            # Fit GP
            gps = get_and_fit_simple_custom_gp(self.X, self.y, self.grads)
            obj_fn_gp, grad_gps = gps
            
            # Optimize acquisition function and get next query point
            bounds = torch.cat([self.obj_fn.low.unsqueeze(0),
                            self.obj_fn.high.unsqueeze(0)]).type(self.dtype)
            
            if self.acq_func == 'EI':
                best_f = torch.max(self.y)
                acq_func = ExpectedImprovement(obj_fn_gp, best_f=best_f)

            candidates = optimize_acq_func_and_get_candidates(
                acq_func=acq_func,
                grad_acq=self.grad_acq,
                bounds=bounds,
                grad_gps=grad_gps,
                order=self.order
            )
            
            if self.order:
                best_candidate = get_next_query_point(obj_fn_gp, candidates)
            else:
                best_candidate = candidates[0]
            
            # Function and gradient evaluation at the new point
            y_new = self.obj_fn.forward(best_candidate).unsqueeze(0)
            grad_new = self.obj_fn.backward()
            best_candidate = best_candidate.unsqueeze(0)
            
            # Update X, y and grads
            self.X = torch.cat([self.X, best_candidate])
            self.y = torch.cat([self.y, y_new])
            self.grads = torch.cat([self.grads, grad_new])
            
        return self.X, self.y, self.grads
        