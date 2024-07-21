from warnings import warn

import torch
from inkstone.backends.GenericBackend import GenericBackend


class TorchBackend(GenericBackend):

    def __init__(self):
        super().__init__()
        self.raw_type = torch.Tensor
        self.abs = torch.abs
        self.sqrt = torch.sqrt
        self.arange = torch.arange
        self.ceil = torch.ceil
        self.where = torch.where
        self.la = torch.linalg
        self.diag = torch.diag
        self.sin = torch.sin
        self.cos = torch.cos
        self.arccos = torch.arccos
        self.arcsin = torch.arcsin
        self.ones = torch.ones
        self.square = torch.square
        self.concatenate = torch.concatenate
        self.exp = torch.exp
        self.sinc = torch.sinc
        self.zeros = torch.zeros
        self.tan = torch.tan
        self.roll = torch.roll
        self.sum = torch.sum
        self.dot = torch.dot
        self.hsplit = torch.hsplit
        self.repeat = torch.repeat_interleave
        self.reshape = torch.reshape
        self.moveaxis = torch.moveaxis
        self.full = torch.full
        self.logical_not = torch.logical_not
        self.maximum = torch.maximum
        self.einsum = torch.einsum
        self.lu_factor = torch.linalg.lu_factor
        #       self.lu_solve = torch.linalg.lu_solve
        self.fft = torch.fft
        self.slogdet = torch.slogdet
        self.solve = torch.linalg.solve
        self.linspace = torch.linspace
        self.eye = torch.eye
        self.conj = torch.conj
        self.cross = torch.cross

        self.pi = torch.pi
        self.float64 = torch.float64  # default float precision
        self.int32 = torch.int32  # defualt int precision
        self.complex128 = torch.complex128  # default complex precision

    def parseData(self, i: any, dtype=None):
        if type(i) is self.raw_type:
            # print(i)
            if dtype is not None and dtype != i.dtype:
                return self.castType(i, dtype)
            else:
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
        return torch.tensor(i, dtype=dtype)

    def meshgrid(self, a, b):
        # Implement PyTorch-specific custom function
        return torch.meshgrid(a, b, indexing='xy')

    def castType(self, i, typ):  # typ(e), avoid collision with keyword
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

    def cross(self, a, b):
        return torch.cross(torch.tensor((a[0], a[1], 0), dtype=torch.float64),
                           torch.tensor((b[0], b[1], 0), dtype=torch.float64), dim=-1)[-1]

    def laCross(self, a, b):
        return torch.linalg.cross(torch.tensor((a[0], a[1], 0), dtype=torch.float64),
                                  torch.tensor((b[0], b[1], 0), dtype=torch.float64), dim=-1)[-1]

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

    def triu_indices(self, row, col=None, offset=0):
        #a: input, b: dim to sort along, c: descending, d: controls the relative order of equivalent elements
        if not col:
            #print("col needs to be specified when using torch. But here it's set =m if missing, like what numpy does")
            col = row
        return torch.triu_indices(row, col, offset)

    def lu_solve(self, p, q):
        return torch.linalg.lu_solve(p[0], p[1], q)

    def __getattr__(self, name):
        if hasattr(torch, name):
            return getattr(torch, name)
        elif hasattr(self, name):
            return getattr(self, name)
        else:
            raise AttributeError(f"'{self.__name__}' has no attribute '{name}'")

    def norm(self, i, dim=None):
        return torch.linalg.norm(i, dim=dim)

    def clone(self, i):
        return i.clone()

    def argsort(self, ipt, dim=-1, c=None, d=None):
        if c or d:
            warn("The 3rd and 4th argument are for different purpose in torch and numpy")
        if c is None: c = False
        if d is None: d = False
        return torch.argsort(ipt, dim=dim, descending=c, stable=d)

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


def getLsDepth(ls):
    if type(ls) is list:
        depth = 1
        tmp = ls[0]
        while type(tmp) is list:
            depth += 1
            tmp = tmp[0]
        return depth
    raise ValueError("Not a list")
