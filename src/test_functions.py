import math
import torch
from ObjectiveFunction import ObjectiveFunction

class LeBranke(ObjectiveFunction):
    '''
        This class implements a simple 1D function from Le and Branke,
        "Bayesian Optimization Searching for Robust Solutions." In
        Proceedings of the 2020 Winter Simulation Conference. The 
        function is defined as follows:
            -0.5(x+1)sin(pi*x**2)

        Note that this function has to be maximised in the domain
        [0.1, 2.1]. The function attains maximum value of 1.43604
        at x=1.87334.
    '''

    def __init__(self, noise_mean=None, noise_variance=None, 
        negate=False):

        dims = 1
        low = torch.tensor([0.1])
        high = torch.tensor([2.1])

        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )

        self.true_opt_value = 1.43604

    def evaluate_true(self, X):
        '''
            This function calculates the value of the Branin function
            without any noise.
            
            For more information, refer to the ObjectiveFunction class
            docs.
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2

        self.f_x = -0.5*(X+1)*torch.sin(torch.pi*X**2)

        return self.f_x.squeeze(-1)


class Branin(ObjectiveFunction):
    '''
        This class implements the Branin function, a simple benchmark
        function in two dimensions.
        
        Branin is usually evaluated on [-5, 10] x [0, 15]. The minima
        of Branin is 0.397887 which is obtained at (-pi, 12.275),
        (pi, 2.275) and (9.42478, 2.475).
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/branin.html
    '''
    def __init__(self, noise_mean=None, noise_variance=None, 
        negate=False):
        
        dims = 2
        low = torch.tensor([-5, 0])
        high = torch.tensor([10, 15])
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.397887
        
    def evaluate_true(self, X):
        '''
            This function calculates the value of the Branin function
            without any noise.
            
            For more information, refer to the ObjectiveFunction class
            docs.
        '''
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        
        self.f_x = ((X[:, 1] - 5.1/(4*torch.pi**2)*X[:, 0]**2 + 
                 5*X[:, 0]/torch.pi - 6)**2 +
                 10*(1- 1/(8*torch.pi))*torch.cos(X[:, 0]) + 10)
                                  
        return self.f_x
        
class Levy(ObjectiveFunction):
    '''
        This class implements the Levy function.
        
        The function is usually evaluated on the hypercube xi belongs
        to [-10, 10] for i=1,...,d. The global minima is 0.0 which is
        obtained at (1,...,1).
        
        For information on Branin function visit:
            https://www.sfu.ca/~ssurjano/levy.html
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        negate=False):
        
        low = torch.tensor([-10]*dims)
        high = torch.tensor([10]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0
        
    def evaluate_true(self, X):
        '''
            This function calculates the value of the Levy function
            without any noise.
            
            For more information, refer to the ObjectiveFunction class
            docs.
            
            Source:
            ------
            https://botorch.org/api/_modules/botorch/test_functions/synthetic.html#Levy
        '''
        w = 1.0 + (X - 1.0) / 4.0
        part1 = torch.sin(math.pi * w[..., 0]) ** 2
        part2 = torch.sum(
            (w[..., :-1] - 1.0) ** 2
            * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0) ** 2),
            dim=-1,
        )
        part3 = (w[..., -1] - 1.0) ** 2 * (
            1.0 + torch.sin(2.0 * math.pi * w[..., -1]) ** 2
        )
        
        self.f_x = part1 + part2 + part3
        
        return self.f_x
        
class Rosenbrock(ObjectiveFunction):
    '''
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        negate=False):
        
        low = torch.tensor([-5]*dims)
        high = torch.tensor([10]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0
        
    def evaluate_true(self, X):
        return torch.sum(
            100.0 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (X[..., :-1] - 1) ** 2,
            dim=-1,
        )

class Ackley(ObjectiveFunction):
    '''
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        negate=False):
        
        low = torch.tensor([-32.768]*dims)
        high = torch.tensor([32.768]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = 0.0

        self.a = 20
        self.b = 0.2
        self.c = 2 * math.pi

    def evaluate_true(self, X):
        a, b, c = self.a, self.b, self.c
        part1 = -a * torch.exp(-b / math.sqrt(self.dims) * torch.norm(X, dim=-1))
        part2 = -(torch.exp(torch.mean(torch.cos(c * X), dim=-1)))
        return part1 + part2 + a + math.e

class Hartmann(ObjectiveFunction):
    '''
        TO-DO: Implement Hartmann for multiple dimensions.
    '''
    def __init__(self, dims, noise_mean=None, noise_variance=None, 
        negate=False):

        if dims not in (3, 4, 6):
            raise ValueError(f"Hartmann with dim {dims} not defined")
        
        low = torch.tensor([0.0]*dims)
        high = torch.tensor([1.0]*dims)
        
        super().__init__(
            dims,
            low,
            high,
            noise_mean=noise_mean,
            noise_variance=noise_variance,
            negate=negate,
        )
        self.true_opt_value = -3.32237  # For 6 dim Hartmann

        self.ALPHA = torch.tensor([1.0, 1.2, 3.0, 3.2])

        if dims == 3:
            A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]]
            P = [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        elif dims == 4:
            A = [
                [10, 3, 17, 3.5],
                [0.05, 10, 17, 0.1],
                [3, 3.5, 1.7, 10],
                [17, 8, 0.05, 10],
            ]
            P = [
                [1312, 1696, 5569, 124],
                [2329, 4135, 8307, 3736],
                [2348, 1451, 3522, 2883],
                [4047, 8828, 8732, 5743],
            ]
        elif dims == 6:
            self.A = torch.tensor([
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ])
            self.P = torch.tensor([
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ])

    def evaluate_true(self, X):
        inner_sum = torch.sum(self.A * (X.unsqueeze(-2) - 0.0001 * self.P) ** 2, dim=-1)
        H = -(torch.sum(self.ALPHA * torch.exp(-inner_sum), dim=-1))
        if self.dims == 4:
            H = (1.1 + H) / 0.839
        return H

if __name__ == "__main__":
    '''
        For testing purposes.
    '''
    le_branke = LeBranke()
    assert round(le_branke.evaluate_true(torch.tensor([[1.87334]])).item(), 4) == 1.4360
    assert round(le_branke.evaluate_true(torch.tensor([[1.23223]])).item(), 5) == 1.11425
    assert round(le_branke.evaluate_true(torch.tensor([[1.58504]])).item(), 5) == -1.29155
    assert round(le_branke.evaluate_true(torch.tensor([[0.734545]])).item(), 6) == -0.860584