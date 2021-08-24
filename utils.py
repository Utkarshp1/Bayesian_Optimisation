import torch
from botorch.optim import optimize_acqf

def generate_initial_data(obj_fn, n, dtype, order):
    '''
        This function generates the initial data for the Bayesian
        Optimisation i.e. the data on which the GP will be fit in the
        first iteration of BO.
        
        Arguments:
        ---------
            - obj_fn: A object of ObjectiveFunction class
            - n: The number of training examples to be generated.
            - dtype: PyTorch dtype
    '''
    X_train = torch.rand(n, obj_fn.dims, dtype=dtype)
    X_train = X_train*obj_fn.high + obj_fn.low
    y_train = obj_fn.forward(X_train)
    if order:
        grads = obj_fn.backward()
        return X_train.detach().clone(), y_train, grads
    else:
        return X_train.detach().clone(), y_train, None
    
def optimize_acq_func_and_get_candidates(acq_func, grad_acq, bounds, grad_gps,
        order=0):
    '''
        This function optimizes the acquisition function and returns the
        points (candidates) where the acquisition function has the
        optimal value.
        
        Arguments:
        ---------
            - acq_func: acquisition function for the objective function.
            - grad_acq: acquisition function for the gradient function
            - bounds: bounds for the domain of the objective function.
            - grad_gps: list of d GPs for the gradient function.
            - order: (int) If 0 then perform 0 order BO else perform 1st
                order BO
                
        Returns:
        -------
            Set of candidates where the acquisition function achieves
            it's optimal value.
    '''
    BATCH_SIZE = 2
    NUM_RESTARTS = 2
    RAW_SAMPLES = 32

    candidates = []
    
    # Optimize the acquisition function for the objective function
    candidate, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
    )
    candidates.append(candidate)
    
    # Optimize the acquisition function for the gradient function if
    # first order Bayesian Optimisation
    if order:
        for grad_gp in grad_gps:
            acq_func = grad_acq(grad_gp)

            candidate, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
            )
            candidates.append(candidate)
        
    return candidates
    
def get_next_query_point(obj_fn_gp, candidates, method="convex", T=1):
    '''
        This function returns the point where the objective function is
        to be queried next.
        
        Arguments:
        ---------
            - obj_fn_gp: (BoTorch Model) This model should be the GP for
                the objective function.
            - candidates: (List of PyTorch tensors) List of possible 
                candidates from which the point to be queried next is to
                be selected.
                
        Returns:
        -------
            Returns a PyTorch tensor of shape (d, ), where d is the 
            dimension of the input taken by the Objective Function.
    '''
    X = torch.stack(candidates).squeeze()
    posterior = obj_fn_gp.posterior(X)
    mean = posterior.mean
    if method == "convex":
        exp_weights = torch.exp(mean/T)
        part1 = exp_weights*X
        part2 = part1/exp_weights.sum()
        return part2.sum(dim=0)
    
    if method == "minimum":
        return X[torch.argmax(mean).item()]

    if method == "best":
        exp_weights = torch.exp(mean/T)
        part1 = exp_weights*X
        part2 = part1/exp_weights.sum()
        part3 = part2.sum(dim=0).unsqueeze(0)

        if obj_fn_gp.posterior(part3).mean.item() > torch.max(mean).item():
            print("convex solution")
            return part3[0, :]
        else:
            return X[torch.argmax(mean).item()]