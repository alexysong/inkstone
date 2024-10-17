import torch
import torch.autograd as autograd
from functools import partial

_T = lambda x: torch.transpose(x, -1, -2)
_dot = partial(torch.einsum, '...ij,...jk->...ik')
_diag = lambda a: torch.eye(a.shape[-1], device=a.device, dtype=a.dtype) * a

def prec_fix(tensor, prec=16):
    # Calculate the scaling factor

    scale = 10 ** prec
    if tensor.dtype == torch.complex128:
        # Handle real and imaginary parts separately
        real = torch.round(tensor.real * scale) / scale
        imag = torch.round(tensor.imag * scale) / scale
        return torch.complex(real, imag)

    return torch.round(tensor * scale) / scale

def _matrix_diag(a):
    reps = torch.tensor(a.shape)
    reps[:-1] = 1
    reps[-1] = a.shape[-1]
    newshape = list(a.shape) + [a.shape[-1]]
    return _diag(torch.tile(a, tuple(reps)).reshape(newshape))


class EigFunction(autograd.Function):
    @staticmethod
    def forward(ctx, A):
        e, v = torch.linalg.eig(A)
        ctx.save_for_backward(A, e, prec_fix(v))
        return e, v

    @staticmethod
    def backward(ctx, grad_e, grad_v):
        A, e, v = ctx.saved_tensors
        n = e.shape[-1]

        ge = _matrix_diag(grad_e)
        f = 1 / (e.unsqueeze(-2) - e.unsqueeze(-1) + 1.e-20)
        f -= _diag(f)

        vt = _T(v)
        r1 = f * _dot(vt, grad_v)
        r2 = -f * (
            _dot(_dot(vt, torch.conj(v)), torch.real(_dot(vt, grad_v)) * torch.eye(n, device=A.device, dtype=v.dtype)))

        # Use torch.linalg.solve instead of explicit inversion
        grad_A = _dot(torch.linalg.solve(vt, ge + r1 + r2), vt)

        if not torch.is_complex(A):
            grad_A = torch.real(grad_A)

        return grad_A


# Wrapper functions
j0 = torch.special.bessel_j0
j1 = torch.special.bessel_j1


#eig = torch.linalg.eig

eig = lambda A: EigFunction.apply(A)
