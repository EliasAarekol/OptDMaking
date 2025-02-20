from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def update_state(self):
        pass

    @abstractmethod
    def get_LP_formulation(self):
        pass

    @abstractmethod
    def lagrange_gradient(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    

