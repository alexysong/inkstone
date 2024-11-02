from warnings import warn
from typing import Type, Dict, Optional
from inkstone.backends.Backend import Backend
import importlib

from pathlib import Path
class BackendRegistry:
    _instance: Optional['BackendRegistry'] = None
    _current_backend: Optional[Backend] = None
    _backend_str: Optional[str] = None
    _backend_types: Dict[str, Type[Backend]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def backend(cls) -> Backend:
        if cls._current_backend is None:
            warn('Backend is not initialized, default to numpy')
            cls.set_backend('numpy')
        return cls._current_backend

    @classmethod
    def get_backend_str(cls) -> str:
        return cls._backend_str

    @classmethod
    def set_backend(cls, backend_name: str) -> Backend:
        """Set the backend by name, with dynamic import functionality"""
        if cls._current_backend is not None:
            warn("You are re-assigning the backend.")

        try:
            # Dynamic import of backend module
            module_path = f'inkstone.backends.{backend_name.capitalize()}Backend'
            class_name = f'{backend_name.capitalize()}Backend'

            try:
                module = __import__(module_path, fromlist=[class_name])
            except ImportError:
                # Alternative import method if the first fails
                module_spec = importlib.util.find_spec(module_path)
                if module_spec is None:
                    raise NotImplementedError(f'{backend_name} is not found')
                module = importlib.util.module_from_spec(module_spec)
                module_spec.loader.exec_module(module)

            BackendClass = getattr(module, class_name)


            cls._current_backend = BackendClass()
            cls._backend_str = backend_name
            print(f"Switched to {backend_name}, {cls._current_backend.raw_type}")
            return cls._current_backend

        except AttributeError:
            raise NotImplementedError(f'{backend_name} backend class not found in module')

    @classmethod
    def reset(cls):
        cls._current_backend = None
        cls._backend_str = None

    @classmethod
    def available_backends(cls) -> list[str]:
        """Get list of available backends by scanning the backends directory"""
        backends_dir = Path(__file__).parent / 'backends'
        backends = []
        if backends_dir.exists():
            for file in backends_dir.glob('*Backend.py'):
                backend_name = file.stem.replace('Backend', '').lower()
                if backend_name != '__init__':
                    backends.append(backend_name)
        return sorted(backends)


# For backward compatibility, create module-level functions
def backend():
    return BackendRegistry.backend()


def set_backend(backend_name: str):
    return BackendRegistry.set_backend(backend_name)