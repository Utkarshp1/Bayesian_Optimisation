import torch
from torch import Tensor
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from botorch.acquisition import AnalyticAcquisitionFunction
from torch.distributions.normal import Normal

class CustomGradientAcquistionFunction(AnalyticAcquisitionFunction):
    '''
        This class implements the acquisition function where we want to
        minimize the expectation of the modulus of the gradient of the
        objective function in a particular direction.
    '''
    def __init__(self, model: Model, maximize: bool = False) -> None:
        ''' 
            We use the AcquisitionFunction constructor, since that of 
            AnalyticAcquisitionFunction performs some validity checks 
            that we don't want here.
            
            Arguments:
            ---------
                - model: A BoTorch Model whose posterior will be used
                    to calculate the expectation.
                - maximize: (boolean) to describe whether we want to
                    minimize or maximize the acquisition function.
        '''
        
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize
        
    def forward(self, X: Tensor) -> Tensor:
        '''
            This function calculates the value of the acquisition
            function at X.
            
            Arguments:
            ---------
                - X: (A PyTorch Tensor) Shape batch_size x q x d, where
                    d is the dimension of the input taken by the 
                    Objective Function
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        posterior = self.model.posterior(X)
        mean = posterior.mean.squeeze()
        covs = posterior.mvn.covariance_matrix.squeeze()
        sigma = torch.sqrt(covs)
        std_normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        pdf_value = (torch.exp(std_normal.log_prob(mean/sigma)))
        acq_value = (2*sigma*(1/torch.sqrt(torch.tensor(2*torch.pi)))*pdf_value 
            + mean*(1-2*std_normal.cdf(-mean/sigma)))
        return acq_value