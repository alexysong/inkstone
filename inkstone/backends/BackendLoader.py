class BackendLoader:
    def __init__(self):
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            raise Exception('Backend is not initialized, please make sure you set the backend before the initialisation of Inkstone')
        return self._backend

    def set_backend(self, backend_name):
        if self._backend is not None:
            raise Exception('Backend can only be set once per lifecycle')

        try:
            module = __import__(f'inkstone.backends.{backend_name.capitalize()}Backend', fromlist=[f'{backend_name.capitalize()}Backend'])
            BackendClass = getattr(module, f'{backend_name.capitalize()}Backend')
            self._backend = BackendClass()
            print(f"Switched to {backend_name}, {self._backend.raw_type}")
        except ImportError:
            raise NotImplementedError(f'{backend_name} is not implemented')

bg = BackendLoader()