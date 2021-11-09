# -*- coding: utf-8 -*-

import numpy as np
# from numpy import linalg as la


def g_pts(num_g, b1, b2):
    """
    given number of lattice points, and two lattice vectors, give the list of all lattice points inside a circle. the total number of lattice points returned is roughly the given number of lattice points.

    This function is indifferent whether b1 and b2 are primitive lattice or reciprocal lattice.

    Parameters
    ----------
    num_g       :   int
                    target total number of lattice points
    b1, b2      :   tuple[float, float]
                    lattice vectors. For rectangular lattice, the input b vectors should be orthogonal

    Returns
    -------
    k_pts       :   list[tuple[float, float]]
                    the k points
    idx         :   list[tuple[int, int]]
                    the indices of the k points, i.e. (m, n) as in m*b1 + n*b2 is the corresponding k points.

    """

    b1 = np.array(b1)
    b2 = np.array(b2)

    bz_are = np.abs(np.cross(b1, b2))  # Brillouin zone area

    k_radi = np.sqrt(num_g * bz_are / np.pi)  # k points within the vertices are to be included

    k_pts = []
    idx = []

    if b1 @ b2 == 0:  # rectangular lattice
        l1 = np.linalg.norm(b1)
        M = np.ceil(k_radi / l1).astype(int)
        l2 = np.linalg.norm(b2)
        N = np.ceil(k_radi / l2).astype(int)
        # m and n array for quarter parallelogram with edges included
        m_a = np.arange(0, M+1)
        n_a = np.arange(0, N+1)
        m_grid, n_grid = np.meshgrid(m_a, n_a)  # quarter parallelogram, with edges included
        # k points coordinates in the quarter parallelogram
        kkx = m_grid * b1[0] + n_grid * b2[0]
        kky = m_grid * b1[1] + n_grid * b2[1]
        k_dis = np.sqrt(kkx ** 2 + kky ** 2)  # 2d array containing distances
        # which k point in the half parallelogram is inside the circle
        id_n, id_m = np.where(k_dis <= k_radi)  # each is a 1d array. containing the m and n index of the points inside. Note numpy default indexing is row for x column for y.
        for i1, i2 in zip(id_n, id_m):
            if m_a[i2] == 0 and n_a[i1] == 0:
                k_pts.append((0., 0.))
                idx.append((0, 0))
            elif m_a[i2] == 0 or n_a[i1] == 0:  # add these points on the two arms of the quarter parallelogram and their opposite points.
                k = (kkx[i1, i2], kky[i1, i2])
                k_i = (-kkx[i1, i2], -kky[i1, i2])  # opposite point
                k_pts.append(k)
                k_pts.append(k_i)
                idx.append((m_a[i2], n_a[i1]))
                idx.append((-m_a[i2], -n_a[i1]))
            else:  # for points not on the two arms, add them and their 3 symmetry partners
                k = (kkx[i1, i2], kky[i1, i2])
                k1 = (-kkx[i1, i2], kky[i1, i2])
                k2 = (kkx[i1, i2], -kky[i1, i2])
                k3 = (-kkx[i1, i2], -kky[i1, i2])
                k_pts += [k, k1, k2, k3]
                idx += [(m_a[i2], n_a[i1]), (-m_a[i2], n_a[i1]), (m_a[i2], -n_a[i1]), (-m_a[i2], -n_a[i1])]
    else:
        l1 = bz_are / np.linalg.norm(b2)
        M = np.ceil(k_radi / l1).astype(int)
        l2 = bz_are / np.linalg.norm(b1)
        N = np.ceil(k_radi / l2).astype(int)
        # m and n array for half parallelogram with edges included
        m_a = np.arange(-M, M+1)
        n_a = np.arange(0, N+1)
        m_grid, n_grid = np.meshgrid(m_a, n_a)  # half parallelogram, with edges included
        # k points coordinates in the half parallelogram
        kkx = m_grid * b1[0] + n_grid * b2[0]
        kky = m_grid * b1[1] + n_grid * b2[1]
        k_dis = np.sqrt(kkx ** 2 + kky ** 2)  # 2d array containing distances
        # which k point in the half parallelogram is inside the circle
        id_n, id_m = np.where(k_dis <= k_radi)  # each is a 1d array. containing the index of m and n of the points inside
        k_pts.append((0., 0.))
        idx.append((0, 0))
        for i1, i2 in zip(id_n, id_m):
            if not (m_a[i2] <= 0 and n_a[i1] == 0):  # excluding this half "arm" to avoid duplication. n index = 0 and m index > 0 is already included.
                k = (kkx[i1, i2], kky[i1, i2])
                k_i = (-kkx[i1, i2], -kky[i1, i2])  # opposite point
                k_pts.append(k)
                k_pts.append(k_i)
                idx.append((m_a[i2], n_a[i1]))
                idx.append((-m_a[i2], -n_a[i1]))

    return k_pts, idx

