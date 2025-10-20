class _TransposeView:
    def __init__(self, original):
        self._original = original
    
    def __matmul__(self, rhs):
        return self._original._rmatvec(rhs)
    
    def dot(self, rhs):
        return self._original._rmatvec(rhs)
    
    @property
    def T(self):
        return self._original

    def getattr(self, name):
        # Delegate any other attribute access to original
        return getattr(self._original, name)
    
class FilterKernel:
    def __init__(self):
        self.weights = None
        self.offsets = None
        self.shape = None
        self.matvec = self.dot
        
    def _matvec(self, rho):
        pass
    
    def _rmatvec(self, rho):
        pass
    
    def dot(self, rho):
        if isinstance(rho, type(self.weights)):
            if rho.ndim == 1:
                if rho.shape[0] == self.shape[1]:
                    return self._matvec(rho)
                else:
                    raise ValueError("Input vector size does not match the filter kernel size.")
            else:
                raise NotImplementedError("Only vector inputs are supported.")
        else:
            raise ValueError(f"Input must be a {type(self.weights)} array vector.")
    
    def __matmul__(self, rhs):
        return self.dot(rhs)
    
    @property
    def T(self):
        return _TransposeView(self)