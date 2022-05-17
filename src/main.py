import torch
from experiments import Experiment

dtype = torch.double
BUDGET = 20

config = {
    "experiment_name": "levy-try",
    "objective_function": "levy",
    "zobo_acq_func": "EI",
    "grad_acq_func": "SumGradientAcquisitionFunction",
    "order": 1,
    "budget": 100,
    "runs": 2,
    "noise_mean": 0.0,
    "noise_variance": 0.01,
    "max/min": "min",
    "dims": 4,
    "query_point_selection": "topk",
    "num_restarts": 2,
    "raw_samples": 32,
    "random_seed": 42,
    "exp_dir": "./",
    "init_examples" : 5
}

experiment = Experiment(is_major_change=False, config=config, dtype=dtype)
experiment.perform_experiment()