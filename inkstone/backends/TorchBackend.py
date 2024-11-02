import warnings
import torch
from inkstone.backends.Backend import Backend
from inkstone.backends.primitives.torch_primitive import j0, j1, eig


class TorchBackend(Backend):

    def __init__(self):
        super().__init__()
        self.raw_type = torch.Tensor
        self.complex128 = torch.complex128  # default complex precision
        self.float64 = torch.float64  # default float precision
        self.int32 = torch.int32  # default int precision
        self.pi = torch.pi

        self.ifftshift = torch.fft.ifftshift

    def abs(self, *args, **kwargs):
        return torch.abs(*args, **kwargs)

    def arange(self, *args, **kwargs):
        return torch.arange(*args, **kwargs)

    def arccos(self, *args, **kwargs):
        return torch.arccos(*args, **kwargs)

    def arcsin(self, *args, **kwargs):
        return torch.arcsin(*args, **kwargs)

    def ceil(self, *args, **kwargs):
        return torch.ceil(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return torch.concatenate(*args, **kwargs)

    def conj(self, *args, **kwargs):
        return torch.conj_physical(*args, **kwargs)

    def cos(self, *args, **kwargs):
        return torch.cos(*args, **kwargs)

    def diag(self, *args, **kwargs):
        return torch.diag(*args, **kwargs)

    def dot(self, *args, **kwargs):
        return torch.dot(*args, **kwargs)

    def einsum(self, *args, **kwargs):
        return torch.einsum(*args, **kwargs)

    def exp(self, *args, **kwargs):
        return torch.exp(*args, **kwargs)

    def eye(self, *args, **kwargs):
        return torch.eye(*args, **kwargs)

    def full(self, *args, **kwargs):
        return torch.full(*args, **kwargs)

    def hsplit(self, *args, **kwargs):
        return torch.hsplit(*args, **kwargs)

    def linspace(self, *args, **kwargs):
        return torch.linspace(*args, **kwargs)

    def logspace(self, *args, **kwargs):
        return torch.logspace(*args, **kwargs)

    def logical_not(self, *args, **kwargs):
        return torch.logical_not(*args, **kwargs)

    def lu_factor(self, *args, **kwargs):
        return torch.linalg.lu_factor(*args, **kwargs)

    def maximum(self, *args, **kwargs):
        return torch.maximum(*args, **kwargs)

    def moveaxis(self, *args, **kwargs):
        return torch.moveaxis(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return torch.repeat_interleave(*args, **kwargs)

    def reshape(self, *args, **kwargs):
        return torch.reshape(*args, **kwargs)

    def roll(self, *args, **kwargs):
        return torch.roll(*args, **kwargs)

    def sin(self, *args, **kwargs):
        return torch.sin(*args, **kwargs)

    def sinc(self, *args, **kwargs):
        return torch.sinc(*args, **kwargs)

    def slogdet(self, *args, **kwargs):
        return torch.slogdet(*args, **kwargs)

    def solve(self, *args, **kwargs):
        return torch.linalg.solve(*args, **kwargs)

    def square(self, *args, **kwargs):
        return torch.square(*args, **kwargs)

    def sqrt(self, *args, **kwargs):
        return torch.sqrt(*args, **kwargs)

    def sum(self, *args, **kwargs):
        return torch.sum(*args, **kwargs)

    def tan(self, *args, **kwargs):
        return torch.tan(*args, **kwargs)

    def where(self, *args, **kwargs):
        return torch.where(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return torch.stack(*args, **kwargs)

    def j0(self, *args, **kwargs):
        return j0(*args, **kwargs)

    def j1(self, *args, **kwargs):
        return j1(*args, **kwargs)

    def eig(self, *args, **kwargs):
        return eig(*args, **kwargs)

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
                #print("Only floating/complex number has gradients, "
                #      "but seems you give int numbers, it will be "
                #      "converted to floating number.")
                dtype = self.float64 if req_grad else self.int32
            elif float in types:
                dtype = self.float64
            elif str in types:
                print("String type detected, no gradient required")
                return i
            elif complex in types:
                dtype = self.complex128
            else:
                print(type(o))
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

    def getLsDepth(self, ls):
        if type(ls) is list:
            depth = 1
            tmp = ls[0]
            while type(tmp) is list:
                depth += 1
                tmp = tmp[0]
            return depth
        raise ValueError("Not a list")

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
                item_depth = self.getLsDepth(item)
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
        if type(idx) is torch.Tensor and idx.dim() > 1:
            warnings.warn("You are indexing with a list while it might be expected to be tuple. "
                          "This may results in unexpected tensor.")
        if type(idx) is not int and len(idx) == 0:
            return a

        mask = torch.zeros_like(a, dtype=torch.bool)
        mask[idx] = True

        if torch.is_tensor(b) and not self.is_broadcastable(a.shape, b.shape):
            # this is a workaround to solve shape mismatch on b and a
            # empty_like: 1) faster than zeros_like, 2) we only care the value on idx.
            val = torch.empty_like(a)
            val[idx] = b

            # Use torch.where to combine 'a' and 'b' based on the mask
            return torch.where(mask, val, a)

        return torch.where(mask, b, a)

    def assignMul(self, a, idx, b):
        return self.indexAssign(a, idx, a * b)

    def isnan(self, a):
        return torch.isnan(a)

    def div(self, a, b):
        if type(a) is not self.raw_type:
            a = self.data(a)
        if type(b) is not self.raw_type:
            b = self.data(b)
        return a.div(torch.where(b == 0, 1e-16, b))

    def _compute_gradient(self, y, x, retain=False):
        """Helper function to compute gradients for a single tensor."""
        if not x.is_leaf:
            warnings.warn("The parameter x is not a leaf tensor, while it should be."
                          "Check its getter/setter and make sure it's not modified by "
                          "any other operations.")
            x.retain_grad()
        if y.dtype is torch.complex128:
            y.real.backward(retain_graph=True)
            r_grad = self.clone(x.grad) if x.grad is not None else None
            x.grad = None

            y.imag.backward(retain_graph=retain)
            i_grad = self.clone(x.grad) if x.grad is not None else None
            x.grad = None
            return {'real': r_grad, 'imag': i_grad}
        else:
            y.backward(retain_graph=retain)
            return x.grad

    def get_gradient(self, y: torch.Tensor, x):
        if type(x) is not list:
            x = [x]
        r = []
        for xi in x:
            if xi.requires_grad:
                ret = []
                retain = True
                elements = y
                if self.getSize(y) == 1:
                    elements = [y]
                    if len(x) == 1:
                        retain = False
                # Process each element
                for elem in elements:
                    ret.append(self._compute_gradient(elem, xi, retain))
                r.extend(ret)
            else:
                print(f"requires_grad of {xi} is not set to True")
        return r
