import torch
from inkstone.backends.GenericBackend import GenericBackend
from inkstone.primitives.torch_primitive import j0, j1, eig


class TorchBackend(GenericBackend):

    def __init__(self):
        super().__init__()
        self.raw_type = torch.Tensor

        self.abs = torch.abs
        self.arange = torch.arange
        self.arccos = torch.arccos
        self.arcsin = torch.arcsin
        self.ceil = torch.ceil
        self.concatenate = torch.concatenate
        self.conj = torch.conj_physical
        self.cos = torch.cos
        self.diag = torch.diag
        self.dot = torch.dot
        self.einsum = torch.einsum
        self.exp = torch.exp
        self.eye = torch.eye
        self.fft = torch.fft
        self.full = torch.full
        self.hsplit = torch.hsplit
        self.la = torch.linalg
        self.linspace = torch.linspace
        self.logspace = torch.logspace
        self.logical_not = torch.logical_not
        self.lu_factor = torch.linalg.lu_factor
        self.maximum = torch.maximum
        self.moveaxis = torch.moveaxis
        self.repeat = torch.repeat_interleave
        self.reshape = torch.reshape
        self.roll = torch.roll
        self.sin = torch.sin
        self.sinc = torch.sinc
        self.slogdet = torch.slogdet
        self.solve = torch.linalg.solve
        self.square = torch.square
        self.sqrt = torch.sqrt
        self.sum = torch.sum
        self.tan = torch.tan
        self.where = torch.where
        self.stack = torch.stack

        self.j0 = j0
        self.j1 = j1
        self.eig = eig

        self.complex128 = torch.complex128  # default complex precision
        self.float64 = torch.float64  # default float precision
        self.int32 = torch.int32  # default int precision
        self.pi = torch.pi

    def data(self, i: any, dtype=None, **kwargs):

        if i is None:
            return i
        req_grad = False
        try:
            req_grad = kwargs['requires_grad']
        except KeyError:
            pass

        if type(i) is self.raw_type:
            if dtype is not None and dtype != i.dtype:
                raise Exception("Do not use this function to change dtype")
            else:
                if not i.requires_grad and req_grad:
                    i.requires_grad = req_grad
                else:
                    # You can breakpoint here to remove redundant parsing
                    pass
                return i
        o = i
        while type(o) == list or type(o) == tuple:
            o = o[0]
        types = [type(o), type(i)]

        if self.raw_type in types:
            return self.parseList(i)

        if not dtype:
            if int in types:
                dtype = self.int32
            elif float in types:
                dtype = self.float64
            elif str in types:
                print("String type detected, no gradient required")
                return i
            elif complex in types:
                dtype = self.complex128
            else:
                dtype = type(o)

        return torch.tensor(i, dtype=dtype, requires_grad=req_grad)

    def cross(self, a, b, dim=None):
        try:
            return torch.linalg.cross(a, b)
        except RuntimeError:
            return torch.linalg.cross(torch.cat([a, torch.tensor([0])]),
                                      torch.cat([b, torch.tensor([0])]), dim=-1)[-1]

    def ones(self, c, dtype=torch.float64):
        return torch.ones(c, dtype=dtype)

    def zeros(self, c, dtype=torch.float64):
        return torch.zeros(c, dtype=dtype)

    def meshgrid(self, *tensors):
        """
        torch.meshgrid(*tensors) currently has the same behavior as calling numpy.meshgrid(*arrays, indexing=’ij’).

        In the future torch.meshgrid will transition to indexing=’xy’ as the default.
       """
        return torch.meshgrid(*tensors, indexing='xy')

    def castType(self, i, typ):
        return i.to(typ)

    def parseList(self, tup, dim=0):
        d = tup
        while len(t := [i[0].unsqueeze(0) for i in d if type(i) is list and len(i) == 1]) != 0:
            d = t
        while len(t := [torch.stack(i) for i in d if type(i) is tuple]) != 0:
            d = t
        tup = d

        for i in range(len(tup)):
            if type(tup[i]) is list:
                tup[i] = self.parseList(tup[i])
                i -= 1
        if type(tup[0]) is not self.raw_type:
            return torch.tensor(tup)
        return torch.stack(tup, dim=dim)

    def getSize(self, i):
        #np.prod(i.size(),dtype=np.int32)
        #torch.Size is different from torch.tensor
        return torch.prod(torch.tensor(list(i.size())), dtype=torch.int32)

    def delete(self, x, idx, axis=None):
        if axis is None:
            # Treat the tensor as flattened 1D
            x = x.flatten()
            axis = 0
            idx = idx.item()

        skip = [i for i in range(x.size(axis)) if i != idx]
        indices = [slice(None) if i != axis else skip for i in range(x.ndim)]
        return x.__getitem__(indices)

    def triu_indices(self, row, offset=0, col=None):
        if not col:
            #print("col needs to be specified when using torch. But here it's set =m if missing, like what numpy does")
            col = row
        idx = torch.triu_indices(row, col, offset)
        return idx[0], idx[1]

    def lu_solve(self, p, q):
        return torch.linalg.lu_solve(p[0], p[1], q)


    def norm(self, i, ord=None, dim=None):
        return torch.linalg.norm(i, ord=ord, dim=dim)

    def clone(self, i):
        return torch.clone(i)

    def argsort(self, ipt, dim=-1, **kwargs):
        descending = kwargs.pop('descending', False)
        stable = kwargs.pop('stable', True)
        return torch.argsort(ipt, dim=dim, descending=descending, stable=stable)

    def sort(self, a, axis=-1, **kwargs):
        des = kwargs.pop('descending', False)
        stable = kwargs.pop('stable', True)
        sorte, indices = torch.sort(a, dim=axis, descending=des, stable=stable)
        return sorte

    # manual implementation of block, at least matches all results on np docs
    def block(self, arr):
        if not isinstance(arr, list):
            return arr
        depth = 0
        for i, item in enumerate(arr):
            if isinstance(item, int):
                arr[i] = torch.tensor([item])
                depth = 1
            elif isinstance(item, list):
                if not item:
                    raise ValueError("Empty lists are not allowed")
                item_depth = getLsDepth(item)
                depth = item_depth if i == 0 else depth
                if item_depth != depth:
                    raise ValueError(f"Inconsistent depth: {item}'s depth is not equal to expected depth {depth}")
            else:
                item_depth = len(item.size())
                depth = item_depth if i == 0 else depth
                if item_depth != depth:
                    raise ValueError(f"Inconsistent depth: {item}'s depth is not equal to expected depth {depth}")

        output = []
        for ar in arr:
            if isinstance(ar, list):
                unsq = [torch.unsqueeze(t, 0) if t.dim() == 1 else t for t in ar]
                output.append(torch.cat(unsq, depth))
            else:
                output.append(ar)

        return torch.cat(output) if depth == 1 else torch.cat(output, 1)

    # from https://stackoverflow.com/questions/24743753/test-if-an-array-is-broadcastable-to-a-shape
    def is_broadcastable(self, shp1, shp2):
        for a, b in zip(shp1[::-1], shp2[::-1]):
            if a == 1 or b == 1 or a == b:
                pass
            else:
                return False
        return True

    def indexAssign(self, a: torch.Tensor, idx: [int, tuple, torch.Tensor], b):
        if type(idx) is not int and len(idx) == 0:
            return a

        mask = torch.zeros_like(a, dtype=torch.bool)
        mask[idx] = True

        if torch.is_tensor(b) and not self.is_broadcastable(a.shape, b.shape):
            # this is a workaround to solve shape mismatch on b and a
            val = torch.zeros_like(a)
            val[idx] = b

            # Use torch.where to combine 'a' and 'b' based on the mask
            return torch.where(mask, val, a)

        return torch.where(mask, b, a)

    def isnan(self, a):
        return torch.isnan(a)

    @staticmethod
    def prec_fix(tensor, prec=15):
        # Calculate the scaling factor

        scale = 10 ** prec
        if tensor.dtype == torch.complex128:
            # Handle real and imaginary parts separately
            real = torch.floor(tensor.real * scale) / scale
            i = torch.floor(tensor.imag * scale)
            imag = i / scale
            return torch.complex(real, imag)

        return torch.floor(tensor * scale) / scale

    def add(self, a, b):
        return a+b

    def sub(self, a, b):
        return self.prec_fix(a - b)

    def mul(self, a, b):
        return a * b

    def div(self, a, b):
        if type(a) is not self.raw_type:
            a = self.data(a)
        if type(b) is not self.raw_type:
            b = self.data(b)
        return self.prec_fix(a.div(torch.where(b == 0, 1e-16, b)))


def getLsDepth(ls):
    if type(ls) is list:
        depth = 1
        tmp = ls[0]
        while type(tmp) is list:
            depth += 1
            tmp = tmp[0]
        return depth
    raise ValueError("Not a list")
