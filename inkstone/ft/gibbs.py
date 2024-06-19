# -*- coding: utf-8 -*-

from GenericBackend import genericBackend as gb
from typing import List, Union, Tuple, Optional
from warnings import warn


def gibbs_corr(ks: List[Union[float, Tuple[float, float], Tuple[float, float, float]]],
               m: Optional[float] = None,
               method: str = 'Gaussian',
               order: float = 1.,
               factor: float = 1.,
               gb=gb) -> List[Union[float, Tuple[float, float], Tuple[float, float, float]]]:
    """
    Calculate the correction factor to mitigate Gibbs phenomenon.
    options are either Lanczos or Gaussian.

    Parameters
    ----------
    ks      :   the list of k points to calculate the correction factors
    m       :   defines the width of the averaging window for both Lanczos and gaussian. For standard Lanczos m is the largest k "+1".
    method  :   {'Gaussian', 'Lanczos'}, the method to use
    order   :   for Lanczos only; the power of the the sigma factor sinc term
    factor  :   if m not specified, can still multiply the calculated m by this factor.

    Returns
    -------
    s       :   the correction coefficients for gibbs

    """

    if len(ks) == 1 and (m is None):
        warn('Only one k point is given, with no m specified, can not calculate the correction factor.', UserWarning)
        s = gb.parseData([1])
    else:
        if m == 0.:
            raise Exception('m can not be zero')

        ksa = gb.parseData(ks)

        # calculate the norms of the k's
        if ksa.ndim == 1:
            kn = gb.abs(ksa)
        else:
            ax = tuple(i for i in gb.arange(ksa.ndim)[1:])  # e.g. for 3d k, ksa is 4d, and this gives (1, 2, 3)
            kn = gb.la.norm(ksa, axis=ax)  # k norm

        # calculate m if not given
        if m is None:
            #knp = np.partition(kn, -2) #purpose?
            knp = gb.sort(kn)
            m = kn.max() + (knp[-1] - knp[-2])#TODO: undefined order, how specify? topk(largest=False)
            if method == 'Gaussian':
                m *= 0.7

        m *= factor
        ma = kn / m

        if method == 'Lanczos':
            s = (gb.sinc(ma)) ** order
        else:
            s = gb.exp(-ma**2)

    return s.tolist()


