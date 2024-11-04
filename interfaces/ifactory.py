from abc import ABC, abstractmethod

class IFactory(ABC):

    @abstractmethod
    def create():
        pass
