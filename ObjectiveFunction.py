import torch

class ObjectiveFunction(torch.nn.Module):
    '''
        This class is an abstract class for an objective function that
        is to be maximised using Bayesian Optimisation.
        
        Source:
        https://botorch.org/api/_modules/botorch/test_functions/base.html
    '''
    
    def __init__(self, noise_mean=0, noise_variance=None, 
        random_state=None, negate=False):
        '''
            Arguments:
            ---------
                - noise_mean: Mean of the normal noise to be added to 
                    the function and gradient values
                - noise_variance: Variance of the normal noise to be
                    added to the function and gradient values
                - random_state: Equivalent PyTorch manual_seed
                - negate: Multiplies the value of the function obtained
                    with -1. Use this to minimize the Objective Function.
                - dims: The number of dimensions in the objective 
                    function. For example, in Branin function, dims=2
        '''
        
        super().__init__()
        self.noise_mean = noise_mean
        self.noise_variance = noise_variance
        self.negate = negate
        self.dims = dims
        
        if random_state:
            torch.manual_seed(random_state)
            
    def evaluate_true(self, X):
        '''
            This function calculates the value of the objective function
            at X without noise. X is a tensor of shape batch_size x d,
            where d is the dimension of the input taken by the Objective
            Function.
        '''
        raise NotImplementedError
        
            
    def forward(self, X, noise=True):
        '''
            This function calculates the value of the Objective 
            Function at X with some Gaussian Noise added. The shape of X
            is batch_size x d, where d is the dimension of the input
            taken by the Objective Function.
            
            Arguments:
            ---------
                - X: A PyTorch tensor of shape (batch_size, d)
                - noise (boolean): Indicates whether noise should be
                    added.
                    
            Returns:
            -------
                - A PyTorch Tensor of shape (batch_size).
        '''
        
        self.X = X.detach().clone()
        
        batch = self.X.ndimension() > 1
        self.X = self.X if batch else self.X.unsqueeze(0)
        
        self.X.requires_grad = True
        if self.X.grad:
            self.X.grad.data.zero_()
        
        self.f_x = self.evaluate_true(X=self.X)
        with torch.no_grad():
            if noise and self.noise_variance is not None:
                self.f_x += (self.noise_variance * torch.randn_like(self.f_x) +
                    self.noise_mean)
                
        if self.negate:
            self.f_x = -self.f_x
            
        return self.f_x if batch else self.f_x.squeeze(0)
        
    def backward(self, noise=True):
        '''
            This function calculates the gradients at X i.e. the points
            where the Objective Function value was calculated during the
            forward pass.
            
            Arguments:
            ---------
                - noise (boolean): Indicates whether noise should be
                    added.
        '''
        
        external_grad = torch.tensor([1.]*self.X.shape[0])
        self.f_x.backward(gradient=external_grad)
        
        self.grads = self.X.grad
        
        if noise and self.noise_variance is not None:
            self.grads += (self.noise_variance * torch.randn_like(self.grads)
                    + self.noise_mean)
        
        return self.grads