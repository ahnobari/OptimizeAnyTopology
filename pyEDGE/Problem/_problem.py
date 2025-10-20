class Problem:
    def __init__(self, *args, **kwargs):
        pass
    
    def init_desvars(self, *args, **kwargs):
        raise NotImplementedError("init_desvars method must be implemented in subclasses.")
    
    def set_desvars(self, *args, **kwargs):
        raise NotImplementedError("set_desvars method must be implemented in subclasses.")
    
    def get_desvars(self, *args, **kwargs):
        raise NotImplementedError("get_desvars method must be implemented in subclasses.")
    
    def f(self, *args, **kwargs):
        raise NotImplementedError("Objective method must be implemented in subclasses.")

    def nabla_f(self, *args, **kwargs):
        raise NotImplementedError("Gradient method must be implemented in subclasses.")
    
    def g(self, *args, **kwargs):
        raise NotImplementedError("Constraints method must be implemented in subclasses.")
    
    def nabla_g(self, *args, **kwargs):
        raise NotImplementedError("Gradient of constraints method must be implemented in subclasses.")
    
    def constraint_map(self, *args, **kwargs):
        raise NotImplementedError("Constraint map method must be implemented in subclasses.")
    
    def ill_conditioned(self, *args, **kwargs):
        return False
    
    def is_terminal(self):
        return True
    
    def N(self, *args, **kwargs):
        raise NotImplementedError("N method must be implemented in subclasses.")
    
    def m(self, *args, **kwargs):
        raise NotImplementedError("m method must be implemented in subclasses.")
    
    def bounds(self, *args, **kwargs):
        raise NotImplementedError("Bounds method must be implemented in subclasses.")
    
    def is_independent(self, *args, **kwargs):
        raise NotImplementedError("is_independent method must be implemented in subclasses.")
    