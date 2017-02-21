"""
UBSPLET/WAVELET.PY

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


from scipy.signal import fftconvolve
from skimage import img_as_float
from skimage.io import imread
from ubsplet.utils import atrous_algorithm, bspline_filters

import numpy as np


def iubsplet2d(image, order='cubic', level=4):
    """
    Applies the 2D isotropic undecimated b-spline wavelet transform in
    an input image.

    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    order : string, optional
        Order of the B-spline filter to be used in the decomposition.
        Accepts the strings 'null' (filter [1]), 'linear' ([1, 2, 1]),
        'cubic' ([1, 4, 6, 4, 1]), 'quintic' ([1, 6, 15, 20, 15, 6, 1]),
        'septic' ([1, 8, 28, 56, 70, 56, 28, 8, 1]), and 'nonic'
        ([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]). Default is
        'cubic'.
    level : int, optional
        Number of decomposition levels to apply. Default is 4.

    Returns
    -------
    approx : (level, N, M) ndarray
        Array containing the low-pass approximation (smoothing)
        coefficients of the input image, for each level.
    detail : (level, N, M) ndarray
        Array containing the detail B-spline wavelet coefficients
        related to the input image, for each level.

    References
    ----------
    .. [1] Starck, J-L., Murtagh, F. and Bertero, M. "Starlet transform
    in Astronomical Data Processing", in Handbook of Mathematical Methods
    in Imaging, pp 2053-2098. Springer, 2015. doi:
    10.1007/978-1-4939-0790-8_34.
    .. [2] de Siqueira, A.F. et al. Jansen-MIDAS: a multi-level
    photomicrograph segmentation software based on isotropic undecimated
    wavelets, 2016.
    .. [3] de Siqueira, A.F. et al. Estimating the concentration of gold
    nanoparticles incorporated on Natural Rubber membranes using Multi-Level
    Starlet Optimal Segmentation. Journal of Nanoparticle Research, 2014,
    16: 2809. doi: 10.1007/s11051-014-2809-0.

    Examples
    --------
    >>> from skimage.data import camera
    >>> from ubsplet.wavelet import iubsplet2d
    >>> image = camera()
    >>> approx, detail = iubsplet2d(image,
                                    order='quintic',
                                    level=3)

    >>> from skimage.data import coins
    >>> from ubsplet.wavelet import iubsplet2d
    >>> image = coins()
    >>> approx, detail = iubsplet2d(image,
                                    order='linear')
    """

    # check if the image is grayscale.
    if image.shape[-1] in (3, 4):
        raise TypeError('Your image seems to be RGB (shape: {0}). Please'
                        'use a grayscale image.'.format(image.shape))

    if type(image) is np.ndarray:
        image = img_as_float(image)
    elif type(image) is str:
        try:
            image = imread(image, as_grey=True)
        except:
            print('Data type not understood. Please check the input'
                  'data.')
            raise

    # allocating space for approximation and detail results.
    row, col = image.shape
    approx, detail = [np.empty((level, row, col)) for n in range(2)]

    # selecting filter h, based on the chosen b-spline order.
    h_filter, _ = bspline_filters(order)

    # mirroring parameter: lower pixel number.
    if (row < col):
        par = row
    else:
        par = col

    aux_aprx = np.pad(image, (par, par), 'symmetric')

    for curr_level in range(level):
        prev_img = aux_aprx
        h_atrous = atrous_algorithm(h_filter, curr_level)

        # obtaining approximation and wavelet detail coefficients.
        aux_aprx = fftconvolve(prev_img,
                               h_atrous.T*h_atrous,
                               mode='same')
        aux_detl = prev_img - aux_aprx

        # mirroring correction.
        approx[curr_level] = aux_aprx[par:row+par, par:col+par]
        detail[curr_level] = aux_detl[par:row+par, par:col+par]

    return approx, detail


def ubsplet2d(image, order='cubic', level=4):
    """
    Applies the 2D undecimated b-spline wavelet transform in an input
    image.

    Parameters
    ----------
    image : (N, M) ndarray
        Grayscale input image.
    order : string, optional
        Order of the B-spline filter to be used in the decomposition.
        Accepts the strings 'null' (filter [1]), 'linear' ([1, 2, 1]),
        'cubic' ([1, 4, 6, 4, 1]), 'quintic' ([1, 6, 15, 20, 15, 6, 1]),
        'septic' ([1, 8, 28, 56, 70, 56, 28, 8, 1]), and 'nonic'
        ([1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]). Default is
        'cubic'.
    level : int, optional
        Number of decomposition levels to apply. Default is 4.

    Returns
    -------
    aprx : (level, N, M) ndarray
        Array containing the low-pass approximation (smoothing)
        coefficients of the input image, for each level.
    horz : (level, N, M) ndarray
        Array containing the horizontal detail B-spline wavelet
        coefficients related to the input image, for each level.
    vert : (level, N, M) ndarray
        Array containing the vertical detail B-spline wavelet
        coefficients related to the input image, for each level.
    diag : (level, N, M) ndarray
        Array containing the diagonal detail B-spline wavelet
        coefficients related to the input image, for each level.

    References
    ----------
    .. [1] Starck, J-L., Fadili, J. and Murtagh, F. The Undecimated
    Wavelet Decomposition and its Reconstruction. IEEE Transactions on
    Image Processing, 2007, 16(2): 297-309. doi:
    10.1109/TIP.2006.887733.

    Examples
    --------
    >>> from skimage.data import camera
    >>> from ubsplet.wavelet import ubsplet2d
    >>> image = camera()
    >>> aprx, horz, vert, diag = ubsplet2d(image,
                                           order='septic',
                                           level=3)

    >>> from skimage.data import moon
    >>> from ubsplet.wavelet import ubsplet2d
    >>> image = moon()
    >>> aprx, horz, vert, diag = ubsplet2d(image, level=6)
    """

    # check if the image is grayscale.
    if image.shape[-1] in (3, 4):
        raise TypeError('Your image seems to be RGB (shape: {0}). Please'
                        'use a grayscale image.'.format(image.shape))

    if type(image) is np.ndarray:
        image = img_as_float(image)
    elif type(image) is str:
        try:
            image = imread(image, as_grey=True)
        except:
            print('Data type not understood. Please check the input'
                  'data.')
            raise

    # allocating space for approximation and detail results.
    row, col = image.shape
    aprx, horz, vert, diag = [np.empty([level, row, col]) for n in range(4)]

    # selecting filters h and g, based on the chosen b-spline order.
    h_filter, g_filter = bspline_filters(order)

    # mirroring parameter: lower pixel number.
    if (row < col):
        par = row
    else:
        par = col

    aux_aprx = np.pad(image, (par, par), 'symmetric')

    for curr_level in range(level):
        prev_img = aux_aprx
        h_atrous = atrous_algorithm(h_filter, curr_level)
        g_atrous = atrous_algorithm(g_filter, curr_level)

        # obtaining approximation and wavelet detail coefficients.
        aux_aprx = fftconvolve(prev_img,
                               h_atrous.T*h_atrous,
                               mode='same')
        aux_horz = fftconvolve(prev_img,
                               g_atrous.T*h_atrous,
                               mode='same')
        aux_vert = fftconvolve(prev_img,
                               h_atrous.T*g_atrous,
                               mode='same')
        aux_diag = fftconvolve(prev_img,
                               g_atrous.T*g_atrous,
                               mode='same')

        # mirroring correction.
        aprx[curr_level] = aux_aprx[par:row+par, par:col+par]
        horz[curr_level] = aux_horz[par:row+par, par:col+par]
        vert[curr_level] = aux_vert[par:row+par, par:col+par]
        diag[curr_level] = aux_diag[par:row+par, par:col+par]

    return aprx, horz, vert, diag
