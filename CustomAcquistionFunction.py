import torch
from botorch.utils import t_batch_mode_transform
from botorch.acquisition import AnalyticAcquisitionFunction

class CustomGradientAcquistionFunction(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        maximize: bool = True,
    ) -> None:
        ''' 
            We use the AcquisitionFunction constructor, since that of 
            AnalyticAcquisitionFunction performs some validity checks 
            that we don't want here
        '''
        
        super(AnalyticAcquisitionFunction, self).__init__(model[0])
        self.gradient_models = model[1]
        self.maximize = maximize
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))
        
    def forward(self, X: Tensor) -> Tensor:
        '''
        '''
        
        posterior = self.model.posterior(X)
        means = posterior.mean.squeeze(dim=-2)
        
        
        