import torch
from botorch.optim import optimize_acqf

def generate_initial_data(obj_fn, n, dtype):
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
    grads = obj_fn.backward()
    return X_train.detach().clone(), y_train, grads 
    
def optimize_acq_func_and_get_candidates(acq_func, grad_acq, bounds, grad_gps,
        order=0):
    '''
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
    
def get_next_query_point(obj_fn_gp, candidates):
    '''
    '''
    X = torch.stack(candidates).squeeze()
    posterior = obj_fn_gp.posterior(X)
    mean = posterior.mean
    return X[torch.argmax(mean).item()]
    