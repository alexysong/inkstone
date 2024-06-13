#TODO: all numbers should be torch.tensor

import torch
import numpy as np
import autograd.numpy as anp
from warnings import warn


class GenericBackend:
    """
    Generic Backend Definer
    
    Define the generic API for different backends such as torch and numpy
    
    You can to do one of the following to change the backend:
    
    1: run GenericBackend.swichTo(backend) to change the global backend
    
    2: manually pass an GenericBackend object to classes/functions
    
    Parameters
    ----------
    backend     :   str

    """
    def __init__(self, backend: str):
        self.backend = backend
        match self.backend:
            case "torch":
                self.raw_type = torch.tensor
                self.abs = torch.abs
                self.sqrt = torch.sqrt
                self.arange = torch.arange
                self.ceil = torch.ceil
                #self.meshgrid = torch.meshgrid #see function meshgrid()
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
                
                self.pi = torch.pi
                self.float64 = torch.float64 #default float precision
                self.int32 = torch.int32    #defualt int precision
                self.complex128 = torch.complex128 #default complex precision
                self.eye = torch.eye
                self.conj = torch.conj

            case "autograd":
                self.raw_type = anp.ndarray
                
                self.abs = anp.abs
                self.sqrt = anp.sqrt
                self.arange = anp.arange
                self.ceil = anp.ceil
                self.where = anp.where
                self.la = anp.linalg
                self.diag = anp.diag
                self.sin = anp.sin
                self.cos = anp.cos
                self.arccos = anp.arccos
                self.arcsin = anp.arcsin
                self.ones = anp.ones
                self.square = anp.square
                self.concatenate = anp.concatenate
                self.conj = anp.conj
                self.exp = anp.exp
                self.sinc = anp.sinc
                self.zeros = anp.zeros
                self.tan = anp.tan
                self.roll = anp.roll
                self.sum = anp.sum
                self.dot = anp.dot
                self.hsplit = anp.hsplit
                self.repeat = anp.repeat
                self.reshape = anp.reshape
                self.moveaxis = anp.moveaxis
                self.full = anp.full
                self.logical_not = anp.logical_not
                self.maximum = anp.maximum
                self.einsum = anp.einsum
                
                self.pi = np.pi
                self.float64 = np.float64
                self.int32 = np.int32
                self.complex128 = np.complex128
                self.eye = anp.eye

            case "numpy":
                self.raw_type = np.ndarray
                
                self.abs = np.abs
                self.sqrt = np.sqrt
                self.arange = np.arange
                self.ceil = np.ceil
               # self.meshgrid = np.meshgrid
                self.where = np.where
                self.la = np.linalg
                self.diag = np.diag
                self.sin = np.sin
                self.cos = np.cos
                self.arccos = np.arccos
                self.arcsin = np.arcsin
                self.ones = np.ones
                self.square = np.square
                self.concatenate = np.concatenate
                self.conj = np.conj
                self.exp = np.exp
                self.sinc = np.sinc
                self.zeros = np.zeros
                self.tan = np.tan
                self.roll = np.roll
                self.sum = np.sum
                self.dot = np.dot
                self.hsplit = np.hsplit
                self.repeat = np.repeat
                self.reshape = np.reshape
                self.moveaxis = np.moveaxis
                self.full = np.full
                self.logical_not = np.logical_not
                self.maximum = np.maximum
                self.einsum = np.einsum
                
                self.pi = np.pi
                self.float64 = np.float64
                self.int32 = np.int32
                self.complex128 = np.complex128
                self.eye = np.eye
            
            case _:
                raise NotImplementedError
        
    def inputParser(self, i: any, dtype = None):
        o = i
        if type(i) is self.raw_type:
            dtype = i.dtype
        else:
            while type(o) == list or type(o) == tuple:
                o = o[0]
            if not dtype:
                if type(o) == int:
                    dtype = self.int32
                elif type(o) == float:
                    dtype = self.float64
                elif type(o) == str:
                    print("String type detected, no gradient required")
                    return np.array(i)
                elif type(o) == complex:
                    dtype = self.complex128
                else:
                    print(o)
                    print(f"unrecognised type {dtype}, set to float")
                    dtype = type(o)
        
        match self.backend:
            case "torch":
                # if(type(i) == torch.Tensor):
                #    print(i)
                return torch.tensor(i, dtype=dtype)
            case "autograd":
                return anp.array(i, dtype=dtype)
            case "numpy":
                return np.array(i, dtype=dtype)
            case _:
                raise NotImplementedError 

    def meshgrid(self, a, b):
        match self.backend:
            case "torch":
                #default index=ij, while python default index=xy
                #note: future pytorch will change default index to xy to match numpy's behavior
                return torch.meshgrid(a,b, indexing='xy')
            case "autograd":
                return anp.meshgrid(a,b)
            case "numpy":
                return np.meshgrid(a,b)
            case _:
                raise NotImplementedError

    def castType(self, i, typ): #typ(e), avoid collision with keyword
        match self.backend:
            case "torch":
                return i.to(typ)
            case "autograd":
                # potentially no astype methods or methods allowed in autograd
                return i.astype(typ)
            case "numpy":
                return i.astype(typ)
            case _:
                raise NotImplementedError

    def cross(self, a,b):
        match self.backend:
            case "torch":
                #note: torch.cross' dim behavior will be changed to match torch.linalg.cross in future torch
                #by default, torch.cross find first dim that with 3 elements, while torch.la.cross dim=-1
                #in practice, it just expect the vectors where each with size of 3
                return torch.cross(torch.tensor((a[0],a[1],0), dtype=torch.float64),
                                    torch.tensor((b[0],b[1],0), dtype=torch.float64), dim=-1)[-1]
            case "autograd":
                return anp.cross(a,b)
            case "numpy":
                return np.cross(a,b)
            case _:
                raise NotImplementedError
        
        
        
    def laCross(self,a1,a2):
        match self.backend:
            case "torch": 
                return torch.linalg.cross(torch.tensor([a1[0], a1[1], 0.0], dtype=torch.float64), 
                                        torch.tensor([a2[0], a2[1], 0.0], dtype=torch.float64))[-1]
            case "autograd":
                return anp.linalg.cross(a1,a2)
            case "numpy":
                return np.linalg.cross(a1,a2)
            case _:
                raise NotImplementedError
    
    def getSize(self, i):
        match self.backend:
            case "torch":
                #np.prod(i.size(),dtype=np.int32) 
                #torch.Size is different from torch.tensor 
                return torch.prod(torch.tensor(list(i.size())),dtype=torch.int32)
            case "autograd":
                return i.size
            case "numpy":
                return i.size
            case _:
                raise NotImplementedError
        
    def delete(self, x, idx, axis=None):
        match self.backend:
            case "torch":
                if axis is None:
                    # Treat the tensor as flattened 1D
                    x = x.flatten()
                    axis = 0
                    idx = idx.item()

                skip = [i for i in range(x.size(axis)) if i != idx]
                indices = [slice(None) if i != axis else skip for i in range(x.ndim)]
                return x.__getitem__(indices)
            case "autograd":
                return anp.delete(x,idx, axis=axis)
            case "numpy":
                return np.delete(x,idx, axis=axis)
            case _:
                raise NotImplementedError
        
    def clone(self, i, keep_grad=False):
        match self.backend:
            case "torch":
                return i.clone() if keep_grad else i.detach().clone()
            case "autograd":
                anp.copy()
                return i.copy()
            case "numpy":#copy.deepcoy
                np.copy()
                return i.copy()
            case _:
                raise NotImplementedError
            
    
    def triu_indices(self, row, col = None, offset = 0):
        match self.backend:
            case "torch": #a: input, b: dim to sort along, c: descending, d: controls the relative order of equivalent elements
                if not col:
                    print("col needs to be specified when using torch. But here it's set =m if missing, like what numpy does")
                    col = row
                return torch.triu_indices(row,col,offset)
            case "autograd":#a: input, b: axis, c: algorithm, d: order of comparing
                if not col:
                    col = row
                return anp.triu_indices(row,offset,col)
            case "numpy":#a: input, b: axis, c: algorithm, d: order of comparing
                if not col:
                    col = row
                return np.triu_indices(row,offset,col)
            case _:
                raise NotImplementedError
    
    
    
    def argsort(self, a, b=-1, c=None, d=None):
        if c or d:
            warn("The 3rd and 4th argument are for different purpose in torch and numpy")
        match self.backend:
            case "torch":
                return torch.argsort(a,b,c,d)
            case "autograd":
                return anp.argsort(a,b,c,d)
            case "numpy":
                return np.argsort(a,b,c,d)
            case _:
                raise NotImplementedError
            
    def sort(self,i,dim=-1,des=False,sort_alg='quicksort'):
        match self.backend:
            case "torch":
                return torch.sort(i,dim,des)
            case "autograd":
                return anp.sort(i,dim,sort_alg)
            case "numpy":
                return np.sort(i,dim,sort_alg)
            case _:
                raise NotImplementedError
    
    def linspace(self,start, end, num=50, required_grad=False):
        match self.backend:
            case "torch":
                return torch.linspace(start,end,num,requires_grad=required_grad)
            case "autograd":
                return anp.linspace(start,end,num)
            case "numpy":
                return np.linspace(start,end,num)
            case _:
                raise NotImplementedError
    
  #  def partition(self, i, kth, dim=-1):
 #       match self.backend:
   #         case "torch":
 #               return torch.topk(i,kth,dim)
#            case "autograd":
  #              return anp.partition(i,kth,dim)
#            case "numpy":
  #              return np.partition(i,kth,dim)
        
    
    def block(self,arr):
        match self.backend:
            case "torch":
                if type(arr) is list:
                    depth = 0
                    if type(arr[0]) is int:
                        arr[0] = torch.tensor([arr[0]])
                        depth = 1
                    elif type(arr[0]) is list:
                        depth = getLsDepth(arr[0])
                    else:
                        depth = len(arr[0].size())
                    for i in range(1,len(arr)):
                        if type(arr[i]) is int:
                            arr[i] = torch.tensor([arr[i]])
                        elif type(arr[i]) is list:
                            if getLsDepth(arr[i]) != depth:
                                raise ValueError(f"inconsistent depth, {arr[i]}'s depth is not equal to expected depth {depth}")
                        elif len(arr[i].size())!=depth:
                            raise ValueError(f"inconsistent depth, {arr[i]}'s depth is not equal to expected depth {depth}")
                        if type(arr[i]) is list and (arr[i] == []):
                            raise ValueError("there should not have empty list")

                output = []
                for ar in arr:
                    if type(ar) is list:
                        out = []
                        for a in ar:
                            if len(a.size()) == 1:
                                out.append(torch.unsqueeze(a,0))
                            else:
                                out.append(a)
                        print(out)
                        output.append(torch.cat(out,depth))
                    else:
                        output.append(ar)
                if depth == 1:
                    return torch.cat(output)
                else:
                    return torch.cat(output,1)
            case "autograd":
                return anp.block(arr)
            case "numpy":
                return np.block(arr)
            case _:
                raise NotImplementedError
        
            
global genericBackend
genericBackend = GenericBackend("numpy")
def switchTo(backend):
    global genericBackend
    genericBackend = GenericBackend(backend)
    
def getLsDepth(ls):
    if type(ls) is list:
        depth = 1
        tmp = ls[0]
        while(type(tmp) is list):
            depth+=1
            tmp = tmp[0]
        return depth
    raise ValueError("Not a list")
if __name__ == '__main__':
    switchTo("torch")

    A = torch.ones((2, 2))
    B = 2 * A
    #A = torch.tensor([1,2,3])
    #B = torch.tensor([4,5,6])
    #A = torch.eye(2) * 2
    #B = torch.eye(3) * 3
    #print(genericBackend.block([
    #    [A,               torch.zeros((2, 3))],
    #    [torch.ones((3, 2)), B               ]
    #]))
    print(genericBackend.block([A,B]))             # vstack([a, b])
