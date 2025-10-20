class Physx:
    def __init__(self):
        pass
    
    def K(self, x0s):
        raise NotImplementedError("K method must be implemented in subclasses.")
    
    def locals(self, x0s):
        raise NotImplementedError("locals method must be implemented in subclasses.")
    
    def volume(self, x0s):
        raise NotImplementedError("volume method must be implemented in subclasses.")
    
    def neumann(self, x0s):
        raise NotImplementedError("neumann method must be implemented in subclasses.")