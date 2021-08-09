import torch
from Branin import Branin
from CustomAcquistionFunction import CustomGradientAcquistionFunction
from BO import BO

dtype = torch.double
BUDGET = 20

# Initialize the Objective Function
branin = Branin(
    noise_mean=0,
    noise_variance=0.1, 
    random_state=42,
    negate=True
)

bo = BO(
    obj_fn=branin,
    dtype=dtype,
    acq_func='EI',
    grad_acq=CustomGradientAcquistionFunction,
    init_examples=5,
    order=1, 
    budget=BUDGET
)

print(bo.optimize())