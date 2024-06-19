# -*- coding: utf-8 -*-

from typing import List, Tuple, Union
from GenericBackend import genericBackend as gb

def pt_in_poly(vts: List[Tuple[float, float]],
               pt: Union[any, Tuple[float, float]],gb=gb) -> bool:
    """
    Determine if a point is in a polygon

    Parameters
    ----------
    vts     :   vertices of the polygon.
                The first point is also the last, no need to repeat it at the end of the list
    pt      :   coordinate of the point

    Returns
    -------
    yn      :   whether it is inside (True) or not (False)
    """
    yn = False
    x, y = pt
    for i, j in zip(vts, (vts[1:] + [vts[0]])):
        if (i[1] > y) != (j[1] > y) and (x < (i[0] + (j[0]-i[0]) * (y - i[1]) / (j[1] - i[1]))):
            yn = not yn

    return yn


