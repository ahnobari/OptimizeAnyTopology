from ..geom.commons._mesh import Mesh, StructuredMesh

class FiniteElement:
    def __init__(self):
        pass
    
    def add_dirichlet_boundary_condition(self, condition):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def add_neumann_boundary_condition(self, condition):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def add_point_forces(self, forces):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_forces(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_dirichlet_boundary_conditions(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_problem(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_density(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_field(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def solve(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")