from ..Problem._problem import Problem

class Optimizer:
    def __init__(self, problem: Problem, *args, **kwargs):
        self.problem = problem
        self.desvars = problem.init_desvars()
        
    def iter(self, *args, **kwargs):
        raise NotImplementedError("iter method must be implemented in subclasses.")

    def converged(self, *args, **kwargs):
        raise NotImplementedError("converged method must be implemented in subclasses.")
    
    def logs(self, *args, **kwargs):
        raise NotImplementedError("logs method must be implemented in subclasses.")