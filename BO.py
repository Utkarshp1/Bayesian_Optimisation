import torch
from tqdm import tqdm
from utils import generate_initial_data, optimize_acq_func_and_get_candidates, get_next_query_point
from GaussianProcess import get_and_fit_simple_custom_gp
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.utils.transforms import unnormalize, standardize, normalize

class BO:
    '''
        This class implements the Bayesian Optimisation loop.
    '''
    def __init__(self, obj_fn, dtype, acq_func, grad_acq=None, init_examples=5,
        order=0, budget=20):
        '''
            Arguments:
            ---------
                - obj_fn: (Instance of ObjectiveFunction class) The 
                    function which needs to be optimized using Bayesian
                    Optimisation.
                - dtype: A PyTorch data type
                - acq_func: (String) Acquisiton Function to be used for
                    the objective function
                - grad_acq: Acquisiton Function to be used for the 
                    gradient function. If order=0 i.e. Zero order BO is
                    performed then this argument will be ignored.
                - init_examples: (int) Number of points where the 
                    objective function is to be queried.
                - order: (int) If 0 then perform 0 order BO else perform
                    1st order BO
                - budget: (int) Number of times the objective function
                    can be evaluated
        '''
        
        self.obj_fn = obj_fn
        self.dtype = dtype
        self.init_examples = init_examples
        self.acq_func = acq_func
        self.grad_acq = grad_acq
        self.order = order
        self.budget = budget
   
    def optimize(self):
        '''
            This function performs the optimization of the Objective
            Function.
            
            Returns:
            -------
                - X (PyTorch tensor): Containing the X's where the 
                    objective function was queried. Shape: 
                    ((budget + init_examples) x d)
                - y (PyTorch tensor): Containing the value of the 
                    objective function at the corresponding X's.
                    Shape: ((budget + init_examples) x 1)
                - grads (PyTorch tensor): Containing the gradient at the
                    corresponding X's of the objective function. Shape:
                    ((budget + init_examples) x d)
        '''
        
        # Generate initial data for GP fiiting
        self.X, self.y, self.grads = generate_initial_data(
            self.obj_fn,
            n=self.init_examples,
            dtype=self.dtype,
            order=self.order
        )
        
        for i in tqdm(range(self.budget)):
            # Fit GP
            gps = get_and_fit_simple_custom_gp(self.X, self.y, self.grads)
            obj_fn_gp, grad_gps = gps
            
            # Optimize acquisition function and get next query point
            original_bounds = torch.cat([self.obj_fn.low.unsqueeze(0),
                            self.obj_fn.high.unsqueeze(0)]).type(self.dtype)
            # Normalize the bounds
            norm_bounds = torch.stack([self.X.min(dim=0)[0],self.X.max(dim=0)[0]])
            bounds = normalize(original_bounds, bounds=norm_bounds)
            
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
                best_candidate = candidates[0][0]
            
            # Unnormalize the best candidate
            best_candidate = unnormalize(best_candidate, bounds=norm_bounds)
            
            # Function and gradient evaluation at the new point
            y_new = self.obj_fn.forward(best_candidate).unsqueeze(0)
            if self.order:
                grad_new = self.obj_fn.backward()
            best_candidate = best_candidate.unsqueeze(0)
            
            # Update X, y and grads
            self.X = torch.cat([self.X, best_candidate])
            self.y = torch.cat([self.y, y_new])
            if self.order:
                self.grads = torch.cat([self.grads, grad_new])
            
        return self.X, self.y, self.grads
        