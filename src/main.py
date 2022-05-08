import torch
from experiments import Experiment

dtype = torch.double
BUDGET = 20

config = {
    "experiment_name": "levy-try",
    "objective_function": "le_branke",
    "acq_func": "EI",
    "order": 1,
    "budget": 100,
    "runs": 2,
    "noise_mean": 0.0,
    "noise_variance": 0.01,
    "max/min": "max",
    "dims": 1,
    "query_point_selection": "convex",
    "num_restarts": 2,
    "raw_samples": 32,
    "random_seed": 42
}

experiment = Experiment(is_major_change=False, config=config, dtype=dtype)
experiment.perform_experiment()