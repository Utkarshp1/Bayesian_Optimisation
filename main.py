import torch
from Branin import Branin
from utils import generate_initial_data
from GaussianProcess import get_and_fit_simple_custom_gp

dtype = torch.double

branin = Branin(
            dims=2,
            noise_mean=0,
            noise_variance=0.1, 
            random_state=42,
            negate=True
        )
            
generate_initial_data(branin, n=5, dtype=dtype)

gps = get_and_fit_simple_custom_gp(branin.X, branin.f_x, branin.grads)
obj_fn_gp, grad_gps = gps