from abc import ABC, abstractmethod

class Solver(ABC):

    @abstractmethod
    def solve(self,node):
        pass
    

