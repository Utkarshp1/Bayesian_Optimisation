from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

class SimpleCustomGP(ExactGP, GPyTorchModel):
    _num_outputs = 1    # to inform GPyTorchModel API

    def __init__(self, X_train,  y_train):
        # squeeze output dim before passing y_train to ExactGP
        super().__init__(X_train, y_train.squeeze(-1), GaussianLikelihood())
        
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=X_train.shape[-1])
        )
        
        self.to(X_train)    # make sure we're on the right device/dtype

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return MultivariateNormal(mean_x, covar_x)
        
def get_and_fit_simple_custom_gp(X_train, y_train, gradients):
    '''
        This function creates and fits d+1 Gaussian Processes i.e. one 
        for the objective function and one for each direction of the
        gradient.
        
        Arguments:
        ---------
            - X_train: Training examples on which the GP must be fit.
                It must be a PyTorch tensor with shape batch_size x d, 
                where d is the number of features.
            - y_train: The objective function value evaluated at X_train
                plus some noise. It must be a PyTorch tensor of shape 
                batch_size.
            - gradients: Gradient of the objective function calculated 
                at X_train. It must be a PyTorch tensor of shape
                batch_size x d, where d is the number of features.
                
        Returns:
        -------
            A tuple of size 2 where the first element is the GP for the
            objective function and second element is a list containing
            d+1 GPs each fitted along a different dimension of the 
            objective function.
    '''
    function_model = SimpleCustomGP(X_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    gradient_models = []
    gradient_mlls = []
    
    # gradients = _calculate_gradients(X_train)
    
    for i in range(gradients.shape[1]):
        gradient_models.append(SimpleCustomGP(X_train[:, i], gradients[:, i]))
        gradient_mlls.append(
            ExactMarginalLogLikelihood(
                gradient_models[i].likelihood, gradient_models[i]
            )
        )
        fit_gpytorch_model(gradient_mlls[i])
    
    return (function_model, gradient_models)
    
# def _calculate_gradients(X_train):
    # '''
    # '''
    # gradients = []
    # branin = Branin()
    # for x in X_train:
        # params = {"x1": x[0].item(),
                  # "x2": x[1].item()}
        
        # branin.forward(params)
        # branin.backward()
        # gradients.append(branin.gradients)
        
    # return torch.stack(gradients)