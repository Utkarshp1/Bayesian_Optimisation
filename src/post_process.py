import os

import yaml
import torch
import argparse
import matplotlib.pyplot as plt

from experiments import Experiment

def build_obj_func(config : dict):
    experiment = Experiment(is_major_change=False, config=config, 
        dtype=torch.double, mode='eval')

    return experiment.obj_fn

def process_output(y, obj_fn, init_examples, mode='max'):
    if mode=='min':
        cum_res, _ = torch.cummin(y, dim=0)
    elif mode=='max':
        cum_res, _ = torch.cummax(y, dim=0)
    else:
        raise NotImplementedError

    regret = torch.log10(torch.abs(cum_res - obj_fn.true_opt_value))
    mean_regret = regret.mean(dim=-1).detach()[init_examples:, 0]
    std_regret = regret.std(dim=-1).detach()[init_examples:, 0]

    return mean_regret, std_regret

parser = argparse.ArgumentParser(
    description='Script to post_process results of BO experiments')
parser.add_argument('-e','--exps_dir', type=str, required=True, 
    help='''Path to experiments directory where sub-directories contains
        results for different experiments to be compared''')
parser.add_argument('-s', '--per_std', type=float, default=0.1,
    help='Percentage of standard deviation to be plotted as errorbar')
parser.add_argument('-x', '--xlabel', type=str, required=True,
    help='Label for the x-axis of the plot')
parser.add_argument('-y', '--ylabel', type=str, required=True,
    help='Label for the y-axis of the plot')
args = parser.parse_args()

plt.figure()
for exp in os.listdir(args.exps_dir):
    try:
        exp_path = os.path.join(args.exps_dir, exp)

        config_file = os.path.join(exp_path, 'config.yaml')
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        obj_fn = build_obj_func(config)

        y = torch.load(os.path.join(exp_path, 'y.pt'))

        mean_regret, std_regret = process_output(y, obj_fn, 
            config['init_examples'], mode=config['max/min'])
        x_vals = range(1, mean_regret.shape[0] + 1)

        plt.errorbar(x_vals, mean_regret, args.per_std*std_regret, 
            marker='D', label=config['experiment_name'])
    except Exception as e:
        import traceback
        traceback.print_exc()

plt.legend()
plt.xlabel(args.xlabel)
plt.ylabel(args.ylabel)
plt.savefig(os.path.join(args.exps_dir, 'Regret_plot.png'))