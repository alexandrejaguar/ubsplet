"""
UBSPLET/UTILS.PY

Copyright (C) Alexandre Fioravante de Siqueira, 2016

This file is part of ubsplet.

ubsplet is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np


def atrous_algorithm(input_vector, factor=0):
    """
    Applies the a trous (with holes) algorithm in an 1D array. The a
    trous algorithm inserts 2**i-1 zeros between the elements zeros
    between the array elements, according to the chosen factor.

    Parameters
    ----------
    input_vector : array
        1D input array.
    factor : int, optional
        Number of factors to use on the algorithm. Default is 0.

    Returns
    -------
    atrous_vector : array
        The original input array modified by the a trous algorithm,
        according to the desired factor.

    References
    ----------
    .. [1] Holschneider, M., Kronland-Martinet, R., Morlet, J. and
    Tchamitchian P. "A real-time algorithm for signal analysis with the
    help of the wavelet transform", in Wavelets, Time-Frequency Methods
    and Phase Space, pp. 289–297. Springer-Verlag, 1989.
    .. [2] Shensa, M.J. The Discrete Wavelet Transform: Wedding the À
    Trous and Mallat Algorithms. IEEE Transactions on Signal Processing,
    40(10): 2464-2482, 1992. doi: 10.1109/78.157290.
    .. [3] Mallat, S. A Wavelet Tour of Signal Processing (3rd edition).
    Academic Press, 2008.

    Examples
    --------
    >>> from ubsplet.utils import atrous_algorithm
    >>> import numpy as np
    >>> vec1 = np.array([1, 3, 1])
    >>> atrous_vec1 = atrous_algorithm(vec1, factor=2)

    >>> from ubsplet.utils import atrous_algorithm
    >>> import numpy as np
    >>> vec2 = np.array([2, 1])
    >>> atrous_vec2 = atrous_algorithm(vec2, factor=4)
    """

    if factor == 0:
        atrous_vector = np.copy(input_vector)
    else:
        m = input_vector.size
        atrous_vector = np.zeros(m+(2**factor-1)*(m-1))
        # zeroes array depends on vector size and wavelet level.
        k = 0
        for j in range(0, m+(2**factor-1)*(m-1), (2**factor-1)+1):
            atrous_vector[j] = input_vector[k]
            k += 1

    # 2D vectors requires less effort for applying wavelets.
    return np.atleast_2d(atrous_vector)


def bspline_filters(order='cubic'):
    """
    Returns the pair of filters (h, g), where h is the B-spline filter
    chosen by its order and g is the difference between the Kronecker
    delta function and h.

    Parameters
    ----------
    order : string, optional
        Order of the B-spline filter to be used in the decomposition.
        Accepts the strings 'null' (filter [1]), 'linear' ([1, 2, 1]),
        'cubic' ([1, 4, 6, 4, 1]), 'quintic' ([1, 6, 15, 20, 15, 6, 1]),
        'septic' ([1, 8, 28, 56, 70, 56, 28, 8, 1]), and 'nonic'
        ([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]). Default is
        'cubic'.

    Returns
    -------
    h_filter : array
        The B-spline filter, according to order.
    g_filter : array
        The difference between the Kronecker delta function and
        h_filter.

    References
    ----------
    .. [1]
    .. [2]

    Examples
    --------
    >>> from ubsplet.utils import bspline_filters
    >>> h_linear, g_linear = bspline_filters(order='linear')
    >>> h_nonic, g_nonic = bspline_filters(order='nonic')
    """

    order2filter = {
        'null': np.array([1]),
        'linear': np.array([1, 2, 1]),
        'cubic': np.array([1, 4, 6, 4, 1]),
        'quintic': np.array([1, 6, 15, 20, 15, 6, 1]),
        'septic': np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]),
        'nonic': np.array([1, 10, 45, 120, 210, 252, 210, 120,
                           45, 10, 1])
    }
    h_filter = order2filter.get(order, None)
    h_filter = h_filter / h_filter.sum()

    try:
        delta = np.zeros(h_filter.shape)
        delta[get_middleindex(delta)] = 1
        g_filter = delta - h_filter
    except:
        print('Sorry. B-spline order not understood')
        raise

    return h_filter, g_filter


def get_middleindex(input_vector):
    """
    Support function. Helps to determine the central element
    of a filter.

    Input: a filter which size will be determined (filter).
    Output: the index of the middle element.
    """

    return int(np.trunc(len(input_vector) / 2))
