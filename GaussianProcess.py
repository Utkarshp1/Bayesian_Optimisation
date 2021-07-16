from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from Branin import Branin

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
        
def _get_and_fit_simple_custom_gp(Xs, Ys, **kwargs):
    function_model = SimpleCustomGP(Xs[0], Ys[0])
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    
    gradient_models = []
    gradient_mlls = []
    
    gradients = _calculate_gradients(Xs[0])
    
    for i in range(gradients.shape[1]):
        gradient_models.append(SimpleCustomGP(Xs[0], gradients[:, i]))
        mll = ExactMarginalLogLikelihood(
            gradient_models[i].likelihood, gradient_models[i])
        fit_gpytorch_model(mll)
    
    return (function_model, gradient_models)
    
def _calculate_gradients(Xs):
    '''
    '''
    gradients = []
    branin = Branin()
    for x in Xs:
        params = {"x1": x[0].item(),
                  "x2": x[1].item()}
        
        branin.forward(params)
        branin.backward()
        gradients.append(branin.gradients)
        
    return torch.stack(gradients)