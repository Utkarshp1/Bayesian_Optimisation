import torch

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
    y_train = obj_fn.forward(X_train)
    grads = obj_fn.backward()