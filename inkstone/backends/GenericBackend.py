from abc import ABC, abstractmethod


class GenericBackend(ABC):
    """
    Define all the attributes that are used in Inkstone, which needs to be implemented in child backend classes
    Also a generic interface for different implementations
    """

    def __init__(self):
        """
        Define some functions which have the same/compatible API.
        Also claim some constants that will be used (such as pi)
        Incompatible APIs will need to be defined as methods below as generic API
        """
        self.raw_type = None

        self.abs = None
        self.arange = None
        self.arccos = None
        self.arcsin = None
        self.ceil = None
        self.concatenate = None
        self.conj = None
        self.cos = None
        self.diag = None
        self.dot = None
        self.einsum = None
        self.exp = None
        self.eye = None
        self.fft = None
        self.full = None
        self.hsplit = None
        self.la = None
        self.linspace = None
        self.logical_not = None
        self.lu_factor = None
        self.maximum = None
        self.moveaxis = None
        self.repeat = None
        self.reshape = None
        self.roll = None
        self.sin = None
        self.sinc = None
        self.slogdet = None
        self.solve = None
        self.square = None
        self.sqrt = None
        self.sum = None
        self.tan = None
        self.where = None

        self.complex128 = None  # default complex precision
        self.float64 = None  # default float precision
        self.int32 = None  # default int precision
        self.pi = None

    @abstractmethod
    def data(self, i: any, dtype=None, **kwargs):
        """
        Generic data parsing api for user inputs.
        In numpy, it is np.array(); in pytorch, it is torch.tensor()

        Parameters
        ----------
        i:  data with native python data type
        dtype: specified data type

        Returns parsed data
        -------

        """
        pass


    @abstractmethod
    def castType(self, i, typ):
        """
        Generic API for type converting. In numpy, it is np.ndarray.astype(); In pytorch, it is torch.tensor.to()

        Parameters
        ----------
        i: parsed data
        typ: dtype of backend

        Returns
        -------

        """
        pass

    @abstractmethod
    def parseList(self, tup):
        """
        Primarily for torch. For other backends that support native list, it
        works same as data.
        Convert list of tensor to tensor list, e.g. [tensor(1), tensor(2)] => tensor([1,2])

        :param tup: a list of tensor
        :return: tensor list
        """
        pass

    @abstractmethod
    def cross(self, a, b):
        pass

    @abstractmethod
    def ones(self, a, dtype):
        pass

    @abstractmethod
    def zeros(self, a, dtype):
        pass

    @abstractmethod
    def meshgrid(self, *xi):
        """

        Parameters
        ----------
        xi 1-D arrays representing the coordinates of a grid.

        Returns
        -------

        """
        pass

    @abstractmethod
    def getSize(self, i):
        """
        Generic API for getting number of elements in an array-like data
        :param i: arraylike
        :return: number of elements in i
        """
        pass

    @abstractmethod
    def clone(self, i):
        """
        Generic API for deep cloning an array-like data
        Parameters
        ----------
        i   the data to be cloned

        Returns the deep clone of the i
        -------

        """
        pass

    @abstractmethod
    def triu_indices(self, row,  offset=0, col=None):
        if not col:     # the default behavior of numpy when col is not given
            col = row
        pass

    @abstractmethod
    def lu_solve(self, p, q):
        pass

    @abstractmethod
    def norm(self, i, ord=None, dim=None):
        pass


    @abstractmethod
    def argsort(self, ipt, dim=-1, **kwargs):
        pass

    @abstractmethod
    def sort(self,a,axis=-1, **kwargs):
        pass

    @abstractmethod
    def delete(self, x, idx, axis=None):
        pass

    @abstractmethod
    def block(self, arr):
        pass

    @abstractmethod
    def isnan(self, a):
        pass

    def indexAssign(self, a, idx, b):
        """
        For numpy, use index assignment. For differentiation libraries, replace with differentiable version
        """
        a[idx] = b
        return a

    def add(self,a,b):
        return a + b

    def sub(self,a,b):
        return a - b

    def mul(self,a,b):
        return a * b

    def div(self,a,b):
        return a / b

    def assignMul(self, a, idx, b):
        a[idx] = a[idx] * b
        return a

