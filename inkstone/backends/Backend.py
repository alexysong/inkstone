from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Define all the attributes/functions that are used in Inkstone, which needs to be implemented in its child backend classes
    Also a generic interface for different implementations
    """
    def __init__(self):
        self.raw_type = None
        self.complex128 = None  # default complex precision
        self.float64 = None  # default float precision
        self.int32 = None  # default int precision
        self.pi = None

    @abstractmethod
    def abs(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : a list of number or a scalar (as array or tensor, depending on the backend).
        kwargs  : [reserved]

        Returns
        -------
        the absolute value of the input data
        """
        pass

    @abstractmethod
    def arange(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : typically start(number)=0, end(number), step(number)=1
        kwargs  : typically has dtype (default to type of input)

        Returns
        -------
        Return evenly spaced values within a given interval as array or tensor (depending on the backend)
        """
        pass

    @abstractmethod
    def arccos(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : typically one scalar or a list/array/tensor of multiple numbers
        kwargs  : [reserved]

        Returns
        -------
        The inverse of cos (element-wise) so that, if y = cos(x), then x = arccos(y).

        """
        pass

    @abstractmethod
    def arcsin(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : typically one scalar or a list/array/tensor of multiple numbers
        kwargs  : [reserved]

        Returns
        -------
        The inverse of sin (element-wise) so that, if y = sin(x), then x = arcsin(y).

        """
        pass

    @abstractmethod
    def ceil(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : typically one scalar or a list/array/tensor of multiple numbers
        kwargs  : [reserved]

        Returns
        -------
        The ceiling (rounding floating point part up to 1) of the input, element-wise

        """
        pass

    @abstractmethod
    def concatenate(self, *args, **kwargs):
        """
        Join a sequence of arrays along an existing axis.

        Parameters
        ----------
        args    : a1,a2,...(a sequence of array_like)
        kwargs  : typically has dim(axis)=0 and dtype=None

        Returns
        -------
        The concatenated array.

        """
        pass

    @abstractmethod
    def conj(self, *args, **kwargs):
        """
        Return the complex conjugate, element-wise.
        The complex conjugate of a complex number is obtained by changing the sign of its imaginary part

        Parameters
        ----------
        args       : an array_like or scalar input with complex data type
        kwargs     : [reserved]

        Returns
        -------
        The complex conjugate, element-wise. Return scalar if input is a scalar.
        The complex conjugate of a complex number is obtained by changing the sign of its imaginary part
        """
        pass

    @abstractmethod
    def cos(self, *args, **kwargs):
        """

        Parameters
        ----------
        args    : typically one scalar or a list/array/tensor of multiple numbers
        kwargs  : [reserved]

        Returns
        -------
        The cosine of the input data, element-wise.
        """
        pass

    @abstractmethod
    def diag(self, *args, **kwargs):
        """
        Extract a diagonal or construct a diagonal array.

        Parameters
        ----------
        v : array_like
            If v is a 2-D array, return a copy of its diagonal.
            If v is a 1-D array, return a 2-D array with v on the diagonal.
        k : int, optional
            Diagonal in question. The default is 0.
            Use k>0 for diagonals above the main diagonal,
            and k<0 for diagonals below the main diagonal.

        Returns
        -------
        out :
            The extracted diagonal or constructed diagonal array.
        """
        pass

    @abstractmethod
    def dot(self, *args, **kwargs):
        """
        Dot product of two arrays.

        Parameters
        ----------
        a, b : array_like
            Input arrays.

        Returns
        -------
        output : ndarray
            Returns the dot product of a and b.
            If a is an N-D array and b is an M-D array (where M>=2),
            it is a sum product over the last axis of a and the second-to-last axis of b.
        """
        pass

    @abstractmethod
    def einsum(self, *args, **kwargs):
        """
        Einstein summation convention on arrays.

        Parameters
        ----------
        subscripts : str
            Specifies the subscripts for summation as comma separated list of subscript labels.
        operands : list of array_like
            These are the arrays for the operation.

        Returns
        -------
        output : ndarray
            The calculation based on the Einstein summation convention.
        """
        pass

    @abstractmethod
    def exp(self, *args, **kwargs):
        """
        Calculate the exponential of all elements in the input array.

        Parameters
        ----------
        x : array_like
            Input values.

        Returns
        -------
        out : ndarray
            Element-wise exponential of x.
        """
        pass

    @abstractmethod
    def eye(self, *args, **kwargs):
        """
        Return a 2-D array with ones on the diagonal and zeros elsewhere.

        Parameters
        ----------
        N : int
            Number of rows in the output.
        M : int, optional
            Number of columns in the output. If None, defaults to N.
        k : int, optional
            Index of the diagonal: 0 (the default) refers to the main diagonal,
            a positive value refers to an upper diagonal,
            and a negative value to a lower diagonal.
        dtype : data-type, optional
            Data-type of the returned array.

        Returns
        -------
        I : ndarray
            An array with ones on the k-th diagonal and zeros elsewhere.
        """
        pass

    @abstractmethod
    def full(self, *args, **kwargs):
        """
        Return a new array of given shape and type, filled with fill_value.

        Parameters
        ----------
        shape : int or sequence of ints
            Shape of the new array.
        fill_value : scalar
            Fill value.
        dtype : data-type, optional
            The desired data-type for the array.

        Returns
        -------
        out : ndarray
            Array of fill_value with the given shape and dtype.
        """
        pass

    @abstractmethod
    def hsplit(self, *args, **kwargs):
        """
        Split an array into multiple sub-arrays horizontally (column-wise).

        Parameters
        ----------
        ary : ndarray
            Array to be divided into sub-arrays.
        indices_or_sections : int or 1-D array
            If an integer, specifies the number of equally shaped sub-arrays.
            If a 1-D array, indicates the indices where the splits should occur.

        Returns
        -------
        sub-arrays : list of ndarrays
            A list of sub-arrays.
        """
        pass

    @abstractmethod
    def linspace(self, *args, **kwargs):
        """
        Return evenly spaced numbers over a specified interval.

        Parameters
        ----------
        start : scalar
            The starting value of the sequence.
        stop : scalar
            The final value of the sequence.
        num : int, optional
            Number of samples to generate. Default is 50.
        endpoint : bool, optional
            If True, stop is the last sample. Otherwise, it is not included.
            Default is True.
        dtype : dtype, optional
            The type of the output array.

        Returns
        -------
        samples : ndarray
            There are num equally spaced samples in the closed interval [start, stop].
        """
        pass

    @abstractmethod
    def logical_not(self, *args, **kwargs):
        """
        Compute the truth value of NOT x element-wise.

        Parameters
        ----------
        x : array_like
            Logical NOT is applied to the elements of x.

        Returns
        -------
        out : ndarray
            Boolean result of the logical NOT operation applied to the elements of x.
        """
        pass

    @abstractmethod
    def lu_factor(self, *args, **kwargs):
        """
        Compute pivoted LU decomposition of a matrix.

        Parameters
        ----------
        a : array_like, shape (M, M)
            Matrix to decompose.
        overwrite_a : bool, optional
            Whether to overwrite data in a (may improve performance).
        check_finite : bool, optional
            Whether to check that the input matrix contains only finite numbers.

        Returns
        -------
        lu : ndarray, shape (M, M)
            Matrix containing U in its upper triangle,
            and L in its lower triangle.
        piv : ndarray, shape (M,)
            Pivot indices representing the permutation matrix P.
        """
        pass

    @abstractmethod
    def maximum(self, *args, **kwargs):
        """
        Element-wise maximum of array elements.

        Parameters
        ----------
        x1, x2 : array_like
            The arrays holding the elements to be compared.

        Returns
        -------
        out : ndarray
            The maximum of x1 and x2, element-wise.
        """
        pass

    @abstractmethod
    def moveaxis(self, *args, **kwargs):
        """
        Move axes of an array to new positions.

        Parameters
        ----------
        a : ndarray
            The array whose axes should be reordered.
        source : int or sequence of int
            Original positions of the axes to move.
        destination : int or sequence of int
            Destination positions for each of the original axes.

        Returns
        -------
        result : ndarray
            Array with moved axes.
        """
        pass

    @abstractmethod
    def repeat(self, *args, **kwargs):
        """
        Repeat elements of an array.

        Parameters
        ----------
        a : array_like
            Input array.
        repeats : int or array of ints
            The number of repetitions for each element.
        axis : int, optional
            The axis along which to repeat values.

        Returns
        -------
        repeated_array : ndarray
            Output array which has the same shape as a, except along
            the given axis.
        """
        pass

    @abstractmethod
    def reshape(self, *args, **kwargs):
        """
        Gives a new shape to an array without changing its data.

        Parameters
        ----------
        a : array_like
            Array to be reshaped.
        newshape : int or tuple of ints
            The new shape should be compatible with the original shape.

        Returns
        -------
        reshaped_array : ndarray
            Array with the new shape.
        """
        pass

    @abstractmethod
    def roll(self, *args, **kwargs):
        """
        Roll array elements along a given axis.

        Parameters
        ----------
        a : array_like
            Input array.
        shift : int or tuple of ints
            The number of places by which elements are shifted.
        axis : int or tuple of ints, optional
            Axis or axes along which elements are shifted.

        Returns
        -------
        res : ndarray
            Output array, with the same shape as a.
        """
        pass

    @abstractmethod
    def sin(self, *args, **kwargs):
        """
        Trigonometric sine, element-wise.

        Parameters
        ----------
        x : array_like
            Angle, in radians.

        Returns
        -------
        y : ndarray
            The sine of each element of x.
        """
        pass

    @abstractmethod
    def sinc(self, *args, **kwargs):
        """
        Return the sinc function.

        Parameters
        ----------
        x : array_like
            Input array.

        Returns
        -------
        out : ndarray
            Output array of same shape as x.
            sinc(x) = sin(πx)/(πx) where x is not 0, and 1 at x = 0.
        """
        pass

    @abstractmethod
    def slogdet(self, *args, **kwargs):
        """
        Compute the sign and logarithm of the determinant of an array.

        Parameters
        ----------
        a : array_like, shape (M, M)
            Input array, must be square 2-D array.

        Returns
        -------
        sign : ndarray
            A number representing the sign of the determinant.
        logdet : ndarray
            The natural log of the absolute value of the determinant.
        """
        pass

    @abstractmethod
    def solve(self, *args, **kwargs):
        """
        Solve a linear system of equations.

        Parameters
        ----------
        a : array_like, shape (M, M)
            Coefficient matrix.
        b : array_like, shape (M,) or (M, N)
            Ordinate or "dependent variable" values.

        Returns
        -------
        x : ndarray, shape (M,) or (M, N)
            Solution to the system a x = b.
        """
        pass

    @abstractmethod
    def square(self, *args, **kwargs):
        """
        Return the element-wise square of the input.

        Parameters
        ----------
        x : array_like
            Input data.

        Returns
        -------
        out : ndarray
            Element-wise x*x.
        """
        pass

    @abstractmethod
    def sqrt(self, *args, **kwargs):
        """
        Return the non-negative square root of an array, element-wise.

        Parameters
        ----------
        x : array_like
            Values whose square roots are required.

        Returns
        -------
        y : ndarray
            An array of the same shape as x, containing the square root of each element.
        """
        pass

    @abstractmethod
    def sum(self, *args, **kwargs):
        """
        Sum of array elements over a given axis.

        Parameters
        ----------
        a : array_like
            Elements to sum.
        axis : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed.
        dtype : dtype, optional
            The type of the returned array and of the accumulator in which the elements are summed.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left in the result as dimensions with size one.

        Returns
        -------
        sum_along_axis : ndarray
            An array with the same shape as a, with the specified axis removed.
        """
        pass

    @abstractmethod
    def tan(self, *args, **kwargs):
        """
        Compute tangent element-wise.

        Parameters
        ----------
        x : array_like
            Input array in radians.

        Returns
        -------
        y : ndarray
            The corresponding tangent values.
        """
        pass

    @abstractmethod
    def where(self, *args, **kwargs):
        """
        Return elements chosen from x or y depending on condition.

        Parameters
        ----------
        condition : array_like, bool
            Where True, yield x, where False, yield y.
        x, y : array_like
            Values from which to choose. x, y and condition need to be broadcastable.

        Returns
        -------
        out : ndarray
            An array with elements from x where condition is True, and elements from y elsewhere.
        """
        pass

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
    def triu_indices(self, row, offset=0, col=None):
        """
        Return the indices for the upper-triangle of an (row, col) array.

        Parameters
        ----------
        row : int
            The number of rows in the arrays for which the returned indices will be valid.
        offset : int, optional
            How far to start from the main diagonal.
            The default is 0, which includes the main diagonal.
            Positive values exclude diagonals above the main diagonal,
            and negative values include diagonals below the main diagonal.
        col : int, optional
            The column dimension of the arrays for which the returned arrays will be valid.
            If None (default), col = row.

        Returns
        -------
        inds : tuple of arrays
            The indices for the triangle. The returned tuple contains two arrays,
            each with the indices along one dimension of the array.
        """
        if not col:
            col = row  # the default behavior of numpy when col is not given
        pass

    @abstractmethod
    def lu_solve(self, p, q):
        """
        Solve an equation system using the LU decomposition.

        Parameters
        ----------
        p : tuple
            LU factorization of a matrix as returned by lu_factor.
            Contains (lu, piv), where lu is the LU decomposition and piv is the pivot indices.
        q : array_like
            Right-hand side of the equation system to be solved.

        Returns
        -------
        x : ndarray
            Solution to the equation system.
        """
        pass

    @abstractmethod
    def norm(self, i, ord=None, dim=None):
        """
        Matrix or vector norm.

        Parameters
        ----------
        i : array_like
            Input array. If dim is None, i must be 1-D or 2-D.
        ord : {non-zero int, inf, -inf, 'fro', 'nuc'}, optional
            Order of the norm:
            inf means numpy's inf object.
            'fro' means Frobenius norm.
            'nuc' means nuclear norm.
        dim : int, tuple of ints, None, optional
            If dim is an int, it specifies which axis of i to compute the vector norm over.
            If dim is a tuple, it specifies the axes that hold 2-D matrices.
            If dim is None, either a vector norm (when i is 1-D) or a matrix norm
            (when i is 2-D) is returned.

        Returns
        -------
        n : float or ndarray
            Norm of the matrix or vector(s).
        """
        pass

    @abstractmethod
    def argsort(self, ipt, dim=-1, **kwargs):
        """
        Returns the indices that would sort an array.

        Parameters
        ----------
        ipt : array_like
            Array to sort.
        dim : int, optional
            Axis along which to sort. The default is -1 (the last axis).
        **kwargs : dict
            Additional arguments to be passed to the underlying sort implementation.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sort the array along the specified axis.
            If a is an n-d array, then index_array is an n-d array with the same
            shape as a.
        """
        pass

    @abstractmethod
    def sort(self, a, axis=-1, **kwargs):
        """
        Return a sorted copy of an array.

        Parameters
        ----------
        a : array_like
            Array to be sorted.
        axis : int or None, optional
            Axis along which to sort. The default is -1 (the last axis).
            If None, the flattened array is used.
        **kwargs : dict
            Additional arguments to be passed to the underlying sort implementation.

        Returns
        -------
        sorted_array : ndarray
            Array of the same type and shape as a, with its elements sorted along
            the specified axis.
        """
        pass

    @abstractmethod
    def delete(self, x, idx, axis=None):
        """
        Return a new array with sub-arrays along an axis deleted.

        Parameters
        ----------
        x : array_like
            Input array.
        idx : slice, int or array of ints
            Indicate indices of sub-arrays to remove along the specified axis.
        axis : int, optional
            The axis along which to delete the subarray defined by idx.
            If None, x is flattened before deletion.

        Returns
        -------
        out : ndarray
            A copy of x with the elements specified by idx removed.
            Note that the dimension is reduced by 1 if axis is not None.
        """
        pass

    @abstractmethod
    def block(self, arr):
        """
        Assemble an array from nested lists of blocks.

        Parameters
        ----------
        arr : nested list of array_like or scalars
            Nested list of array_like objects to be assembled into a single array.
            Each element of the nested list can be a scalar, an array_like object,
            or another nested list.

        Returns
        -------
        block_array : ndarray
            The array assembled from the given blocks.
            The dimensionality of the output is equal to the maximum of the
            dimensionalities of all nested lists plus the depth of nesting.
        """
        pass

    @abstractmethod
    def isnan(self, a):
        """
        Test element-wise for NaN and return result as a boolean array.

        Parameters
        ----------
        a : array_like
            Input array.

        Returns
        -------
        out : ndarray, bool
            True where NaN is encountered, false otherwise.
            This is a scalar if a is a scalar.
        """
        pass
    @abstractmethod
    def get_gradient(self, y, x):
        """

        Parameters
        ----------
        y   : the result that will be affected by the change of x
        x   : (one of) the parameter(s) of y

        Returns
        -------
        The gradient of y with respect to x
        """

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

