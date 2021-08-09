import math
import torch
from ObjectiveFunction import ObjectiveFunction

class Branin(ObjectiveFunction):
    '''
        This class implements the Branin function, a simple benchmark
        function in two dimensions.
        
        Branin is usually evaluated on [-5, 10] x [0, 15]. The minima
        of Branin is 0.397887 which is obtained at (-pi, 12.275),
        (pi, 2.275) and (9.42478, 2.475).
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/branin.html
    '''
    def __init__(self, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        
        dims = 2
        low = torch.tensor([-5, 0])
        high = torch.tensor([10, 15])
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        
    def evaluate_true(self, X):
        '''
            This function calculates the value of the Branin function
            without any noise.
            
            For more information, refer to the ObjectiveFunction class
            docs.
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        self.f_x = ((X[:, 1] - 5.1/(4*torch.pi**2)*X[:, 0]**2 + 
                 5*X[:, 0]/torch.pi - 6)**2 +
                 10*(1- 1/(8*torch.pi))*torch.cos(X[:, 0]) + 10)
                                  
        return self.f_x
        
class Levy(ObjectiveFunction):
    '''
        This class implements the Levy function.
        
        The function is usually evaluated on the hypercube xi belongs
        to [-10, 10] for i=1,...,d. The global minima is 0.0 which is
        obtained at (1,...,1).
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/levy.html
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        
        low = torch.tensor([-10]*dims)
        high = torch.tensor([10]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            random_state=random_state,
            negate=negate,
        )
        
    def evaluate_true(self, X):
        '''
            This function calculates the value of the Levy function
            without any noise.
            
            For more information, refer to the ObjectiveFunction class
            docs.
            
            Source:
            ------
            https://botorch.org/api/_modules/botorch/test_functions/synthetic.html#Levy
        '''
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        
        self.f_x = part1 + part2 + part3
        
        return self.f_x