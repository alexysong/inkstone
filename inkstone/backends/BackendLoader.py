"""
The only function is to provide backend to Inkstone. To run the inkstone, you must do

import inkstone.backends.BackendLoader as bl
bg.set_backend('xxx')

before the initialsation of inkstone, where xxx is the name of the backend.
As long as the name of your class is xxxBackend, this loader will be able to find
the corresponding python class of the backend you set via bg.set_backend(), which means
you do not really need to maintain this class
"""

from warnings import warn

_backend = None


def backend():
    global _backend
    if _backend is None:
        warn('Backend is not initialized, default to numpy')
        set_backend('numpy')
    return _backend


def set_backend(backend_name):
    global _backend
    if _backend is not None:
        warn("Notice that this behaviour, the update to the backend, will not be observed by"
             "other modules that had already imported it. So, the result of this operation"
             "may not match your expectation. You may want to use importlib.reload() to rewrite"
             "the way the backend is imported for now")

        #raise Exception('Backend can only be set once per lifecycle, and set before get')

    try:
        module = __import__(f'inkstone.backends.{backend_name.capitalize()}Backend',
                            fromlist=[f'{backend_name.capitalize()}Backend'])
        BackendClass = getattr(module, f'{backend_name.capitalize()}Backend')
        _backend = BackendClass()
        print(f"Switched to {backend_name}, {_backend.raw_type}")
    except ImportError:
        raise NotImplementedError(f'{backend_name} is not found')
