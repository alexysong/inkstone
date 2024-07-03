from abc import ABC, abstractmethod


class GenericBackend(ABC):
    """
    Define all the attributes that are used in Inkstone, which needs to be implemented in child backend classes
    Also a generic interface for different implementations
    """

    def __init__(self):
        """
        Define some functions which normally have the same/compatible API.
        Incompatible APIs will need to be defined as methods below
        """
        self.raw_type = None
        self.float64 = None
        self.int32 = None
        self.complex128 = None
        self.raw_type = None
        self.abs = None
        self.sqrt = None
        self.arange = None
        self.ceil = None
        # self.meshgrid = torch.meshgrid #see function meshgrid()
        self.where = None
        self.la = None
        self.diag = None
        self.sin = None
        self.cos = None
        self.arccos = None
        self.arcsin = None
        self.ones = None
        self.square = None
        self.concatenate = None
        self.exp = None
        self.sinc = None
        self.zeros = None
        self.tan = None
        self.roll = None
        self.sum = None
        self.dot = None
        self.hsplit = None
        self.repeat = None
        self.reshape = None
        self.moveaxis = None
        self.full = None
        self.logical_not = None
        self.maximum = None
        self.einsum = None
        self.lu_factor = None
        self.fft = None
        self.slogdet = None
        self.solve = None
        self.eye = None
        self.conj = None

        self.pi = None
        self.float64 = None  # default float precision
        self.int32 = None  # defualt int precision
        self.complex128 = None  # default complex precision
        self.linspace = None

    @abstractmethod
    def parseData(self, i: any, dtype=None):
        pass

    @abstractmethod
    def castType(self, i, typ):  # typ(e), avoid collision with keyword
        pass

    @abstractmethod
    def parseList(self, tup):
        """
        Convert list of data to data of list, e.g. [tensor(1), tensor(2)] => tensor([1,2])

        :param tup: a list of data
        :return: a data of list
        """
        pass

    @abstractmethod
    def laCross(self, a, b):
        pass

    @abstractmethod
    def meshgrid(self,a,b):
        pass

    @abstractmethod
    def getSize(self, i):
        """
        same as np.ndarray.size
        :param i: arraylike
        :return: number of elements in i
        """
        pass

    @abstractmethod
    def clone(self, i):
        pass

    @abstractmethod
    def triu_indices(self, row, col=None, offset=0):
        if not col:
            col = row
        pass

    @abstractmethod
    def lu_solve(self, p, q):
        pass

    @abstractmethod
    def norm(self, i, dim=None):
        pass

    @abstractmethod
    def argsort(self, ipt, dim=-1, c=None, d=None):
        pass


from inkstone.backends.NumpyBackend import NumpyBackend
from inkstone.backends.TorchBackend import TorchBackend

backends = {
    'numpy': NumpyBackend,
    'torch': TorchBackend
}
genericBackend = NumpyBackend()


def switchTo(str):
    global genericBackend
    genericBackend = backends[str]()
