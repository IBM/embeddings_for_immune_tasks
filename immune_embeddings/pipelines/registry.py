from typing import Callable


class ConstructorRegistry:
    def __init__(self, name_key: str = "name", param_key: str = "params"):
        self.name_key = name_key
        self.param_key = param_key
        self.registry = dict()

    def register(self, name: str, constructor: Callable):
        self.registry[name] = constructor

    def get(self, params):
        return self.registry[params[self.name_key]](**params[self.param_key])

    def entry(self, constructor: Callable):
        self.register(constructor.__name__, constructor)
        return constructor
