import torch
from ax import ParameterType, RangeParameter, SearchSpace

class Branin():
    '''
        This class implements the Branin function, a simple benchmark
        function in two dimensions. This function is defined as follows:
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/branin.html
    '''
    def __init__(self, noise_mean=None, noise_variance=None, 
        random_state=None):
        '''
            Arguments:
            ---------
                - noise_mean: Mean of the normal noise to be added to 
                    the function and gradient values
                - noise_variance: Variance of the normal noise to be
                    added to the function and gradient values
                - random_state: Equivalent PyTorch manual_seed
        '''
        
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        
        # torch.manual_seed(random_state)
        
    def forward(self, params, *args):
        '''
            This method calculates the value of the Branin function.
            
            Arguments:
            ---------
                - params: (dict) A dict of mapping params to their
                    values. The input is similar to Ax SimpleExperiment
                    evaluation_function input format.
                    https://ax.dev/api/core.html#simpleexperiment
        '''
        
        self.x1, self.x2 = params["x1"], params["x2"]
        self.x1 = torch.tensor(self.x1, requires_grad=True)
        self.x2 = torch.tensor(self.x2, requires_grad=True)
        
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        self.y = ((self.x2 - 5.1/(4*torch.pi**2)*self.x1**2 + 
                  5*self.x1/torch.pi - 6)**2 + 
                  10*(1- 1/(8*torch.pi))*torch.cos(self.x1) + 10)
        
        return {'branin': (self.y.item(), 0.0)}
        
    def backward(self):
        '''
            This method calculates the gradient of the Branin 
            function at the points where the function value was
            evaluated during forward pass.
        '''
        self.y.backward()
        
        self.gradients = torch.tensor([
            self.x1.grad,
            self.x2.grad])
            
    def set_sample_space(self, params):
        '''
            This method sets the search space where the objective
            function needs to optimized. The search space is of the
            type Ax SearchSpace.
            
            Arguments:
            ---------
                - params: (dict) A dict containing keys are the name of
                    the parameter and value being a tuple of two values
                    in the format (lower_bound, upper_bound).
                    
                    For example, if we want to optimize the function 
                    with two parameters namely x1 and x2 in the domain
                    x1 in [-5, 10] and x2 in [0, 15], then params 
                    argument would be like:
                        {"x1": (-5, 10),
                         "x2": (0, 15)}
        '''
        parameters = []
        
        for param_name, param_bounds in params.items():
            parameters.append(
                RangeParameter(
                    name=param_name, 
                    parameter_type=ParameterType.FLOAT,
                    lower=param_bounds[0],
                    upper=param_bounds[1]
                )
            )
        
        self.search_space = SearchSpace(parameters=parameters)