import torch
from ObjectiveFunction import ObjectiveFunction

class Branin(ObjectiveFunction):
    '''
        This class implements the Branin function, a simple benchmark
        function in two dimensions. This function is defined as follows:
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/branin.html
    '''
    def __init__(self, noise_mean=None, noise_variance=None, 
        random_state=None, negate=False):
        
        super().__init__(noise_mean, noise_variance, random_state, negate)
                
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