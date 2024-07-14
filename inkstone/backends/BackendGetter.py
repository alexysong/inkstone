from inkstone.backends.NumpyBackend import NumpyBackend
from inkstone.backends.TorchBackend import TorchBackend
from inkstone.backends.AutogradBackend import AutogradBackend


class BackendGetter:
    def __init__(self):
        self._backend = None
        self.backends = {
            'numpy': NumpyBackend,
            'torch': TorchBackend,
            'autograd': AutogradBackend,
        }

    @property
    def backend(self):
        if self._backend is None:
            raise Exception('Backend not initialized')
        return self._backend

    @backend.setter
    def backend(self, s):
        try:
            self._backend = self.backends[s]()
            print(f"Switched to {s}, {self.backend.raw_type}")
        except KeyError:
            raise NotImplementedError(f'{s} is not implemented')


bg = BackendGetter()
