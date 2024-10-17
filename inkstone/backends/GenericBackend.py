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
        it's the operation that is same as np.array() when using numpy as backend, or torch.tensor() when using
        PyTorch as backend.

        if dtype is not specified, it will be automatically inferred from the input dtype. For example,
        the int object will make dtype as gb.int32, float as gb.float64, complex as gb.complex128

        Parameters
        ----------
        i       : the data (expected to be of the native python type)
        dtype   : data type (gb.float64, gb.complex128, etc.)
        kwargs  : optional keyword arguments that only exists in certain backends, e.g. requires_grad in TorchBackend

        Returns
        -------
        the data that the backend is comfortable to deal with. e.g. np.ndarray when use numpy or torch.Tensor when use PyTorch

        """
        pass


    @abstractmethod
    def castType(self, i, typ):
        """
        Convert the dtype of i to type. In numpy, it is equivalent to np.ndarray.astype(); In pytorch, it is equivalent
        to torch.tensor.to()

        Parameters
        ----------
        i: input data
        typ: data type of the output data

        Returns
        -------
        the input with dtype set to typ
        """
        pass

    @abstractmethod
    def parseList(self, tup):
        """
        This function converts list of tensor to tensor, e.g. [tensor(1), tensor(2)] => tensor([1,2])
        Primarily for torch to solve incompatible list creation operation.
        For example, a = [1,2]; b=[3,4]; [a,b] will be [[1,2],[3,4]].
        However, if a=tensor([1,2]), b=tensor([3,4]), [a,b] will be [tensor([1,2]),tensor([3,4])],
        while you may expect tensor([[1,2],[3,4]]) instead.
        For other backends that support native list, it works same as data().

        Parameters
        ----------
        tup : a python native list of tensors

        Returns
        -------
        tensor
        """
        pass

    @abstractmethod
    def cross(self, a, b, dim=None):
        """
        Parameters
        ----------
        a   :   input data
        b   :   another input data
        dim :   the dimension to take the cross-product in

        Returns
        -------
        the cross product of vectors in dimension dim of a and b
        """
        pass

    @abstractmethod
    def ones(self, a, dtype):
        """
        Parameters
        ----------
        a       : a tuple of int that define the shape
        dtype   : data type of the output

        Returns
        -------
        the array filled with the scalar value 1, with the shape defined by a.
        """
        pass

    @abstractmethod
    def zeros(self, a, dtype):
        """
        Parameters
        ----------
        a       : a tuple of int that define the shape
        dtype   : data type of the output

        Returns
        -------
        the array filled with the scalar value 0, with the shape defined by a.
        """

        pass

    @abstractmethod
    def meshgrid(self, *xi):
        """

        Parameters
        ----------
        xi 1-D arrays representing the coordinates of a grid.

        Returns
        -------
        grids of coordinates specified by the 1D inputs in attr:xi
        """
        pass

    @abstractmethod
    def getSize(self, i):
        """
        Parameters
        ----------
        i : array_like

        Returns
        -------
        number of elements in i
        """
        pass

    @abstractmethod
    def clone(self, i):
        """
        Parameters
        ----------
        i   the data to be cloned

        Returns
        -------
        a deep clone of i
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

