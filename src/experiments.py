import os
import json
import torch
import shutil
import matplotlib.pyplot as plt
from BO import BO
from test_functions import *
from CustomAcquistionFunction import CustomGradientAcquistionFunction

class Experiment():
    def __init__(self, is_major_change, config, dtype):
        self.is_major_change = is_major_change
        self.config = config
        self.obj_fn = self._get_appropriate_func()
        self.dtype = dtype
        
        self.init_examples = 5
        base_size = self.config["budget"] + self.init_examples
        self.X = torch.empty((
            base_size,
            self.config["dims"],
            self.config["runs"]
        ))
        self.y = torch.empty((
            base_size,
            1,
            self.config["runs"]
        ))
        self.grads = torch.empty((
            base_size,
            self.config["dims"],
            self.config["runs"]
        ))
        
        self._init_directory_structure()
        
        
    def _init_directory_structure(self):
        '''
            TO-DO: If major change then copy the code as well.
        '''
        try:
            self.exp_dir = "Experiments/" + self.config["experiment_name"]
            os.mkdir(self.exp_dir)
            # shutil.copy("./config.json", self.exp_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
        
        
    def _get_appropriate_func(self):
        if self.config["objective_function"] == 'le_branke':
            return LeBranke(
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
        if self.config["objective_function"] == 'branin':
            return Branin(
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
            
        if self.config["objective_function"] == 'levy':
            return Levy(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'rosenbrock':
            return Rosenbrock(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'ackley':
            return Ackley(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )

        if self.config["objective_function"] == 'hartmann':
            return Hartmann(
                dims=self.config["dims"],
                noise_mean=self.config["noise_mean"],
                noise_variance=self.config["noise_variance"],
                negate=(self.config["max/min"] == "min")
            )
            
        
    def perform_experiment(self):
        for i in range(self.config["runs"]):
            bo = BO(
                obj_fn=self.obj_fn,
                dtype=self.dtype,
                acq_func=self.config["acq_func"],
                grad_acq=CustomGradientAcquistionFunction,
                init_examples=5,
                order=self.config["order"], 
                budget=self.config["budget"]
            )
            
            X, y, grads = bo.optimize()
            
            self.X[:, :, i] = X
            self.y[:, :, i] = y.unsqueeze(dim=-1)
            if self.config["order"]:
                self.grads[:, :, i] = grads
            self.save_results()
        
        if self.config["max/min"] == "min":
            self.y = -self.y
            
        self.plot_results()
        self.save_results()
        
                
    def save_results(self):
        torch.save(self.X, self.exp_dir + '/X.pt')
        torch.save(self.y, self.exp_dir + '/y.pt')
        torch.save(self.grads, self.exp_dir + '/grads.pt')
        
        summary = {}
        summary.update(self.config)
        # summary["mean_regret"] = self.mean_regret
        summary["optimal_value"] = (self.y.min().item() if 
            self.config["max/min"] == "min" else self.y.max().item())
        
        with open(self.exp_dir + "/summary.json", "w") as outfile: 
            json.dump(summary, outfile)
            
        
    def plot_results(self):
        if self.config["max/min"] == "min":
            cum_min, _ = torch.cummin(self.y, dim=0)
            self.regret = torch.log10(
                torch.abs(cum_min - self.obj_fn.true_opt_value)
            )
        else:
            cum_max, _ = torch.cummax(self.y, dim=0)
            self.regret = torch.log10(
                torch.abs(cum_max - self.obj_fn.true_opt_value)
            )
        self.mean_regret = self.regret.mean(dim=-1)
        self.std_regret = self.regret.std(dim=-1)
        
        plt.figure()
        x_vals = range(1, self.mean_regret.shape[0] - 4)
        plt.errorbar(x_vals, self.mean_regret[5:, 0], 0.1*self.std_regret[5:, 0], linestyle='solid', marker='D')
        # plt.plot(self.mean_regret[5:, 0])
        plt.savefig(self.exp_dir + '/Mean_Regret_vs_Num_iterations.png')
        
        