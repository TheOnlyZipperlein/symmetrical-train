from pyparsing import ABC, abstractmethod

class IWrapper(ABC):
    @abstractmethod
    def grid_search_best_functions(self, limits:tuple[int, int, int]):
        pass
    
    @abstractmethod
    def random_search_best_functions(self, limits:tuple[int, int, int], sample_size:int):
        pass

    @abstractmethod
    def gan_improve_functions(self, epochs:int):
        pass

    @abstractmethod
    def evaluate(self, parameter_list:list):
        pass

    @abstractmethod
    def get_predictions(self, context:str):
        pass
