import warnings
import numpy as np
import math
from skimage.transform._geometric import _GeometricTransform

# from skimage.transform._geometric import PolynomialTransform
#
# class PolynomialTransform(PolynomialTransform):
#     def estimate(self, src, dst, order=4):
#         src = src.astype(float)
#         dst = dst.astype(float)
#         return super().estimate(src, dst, order)
#
#     def __call__(self, coords):
#         coords = coords.astype(float)
#         return super().__call__(coords)

class PolynomialTransform(_GeometricTransform):
    def __init__(self, params=None, dimensionality=None):
        if dimensionality is None:
            dimensionality = 2
        elif dimensionality != 2:
            raise NotImplementedError(
                'Polynomial transforms are only implemented for 2D.'
            )
        if params is None:
            # default to transformation which preserves original coordinates
            params = np.array([[0, 1, 0], [0, 0, 1]])
        if params.shape[0] != 2:
            raise ValueError("invalid shape of transformation parameters")
        self.params = params

    def estimate(self, source, destination, order=4):
        source = source.astype(float)
        # A = np.vstack([source[:, 0] ** i * source[:, 1] ** j for i in range(order + 1) for j in range(order + 1)]).T
        A = np.vstack([source[:, 0] ** (j - i) * source[:, 1] ** i for j in range(order+1) for i in range(j+1)]).T
        B = destination.astype(float)

        # Based on numpy.polyfit
        scale = np.sqrt(A * A).sum(axis=0)
        coeff, r, rank, s = np.linalg.lstsq(A / scale, B, rcond=None)
        coeff = (coeff.T / scale).T

        # warn on rank reduction, which indicates an ill conditioned matrix
        if rank != len(coeff):
            msg = "Estimation may be poorly conditioned"
            if int(np.__version__[0]) == 1:
                warnings.warn(msg, np.RankWarning, stacklevel=4)
            else:
                warnings.warn(msg, np.exceptions.RankWarning, stacklevel=4)

        self.params = coeff.T
        # print(rank)
        return True

    def __call__(self, coords):
        # order = int(np.sqrt(len(self.params)))-1
        number_of_parameters = len(self.params.ravel())
        order = int((- 3 + math.sqrt(9 - 4 * (2 - number_of_parameters))) / 2)
        coords = coords.astype(float)
        # A = np.vstack([coords[:, 0] ** i * coords[:, 1] ** j for i in range(order + 1) for j in range(order + 1)]).T
        A = np.vstack([coords[:, 0] ** (j - i) * coords[:, 1] ** i for j in range(order + 1) for i in range(j + 1)]).T
        return A@self.params.T

    @property
    def inverse(self):
        raise NotImplementedError(
            'There is no explicit way to do the inverse polynomial '
            'transformation. Instead, estimate the inverse transformation '
            'parameters by exchanging source and destination coordinates,'
            'then apply the forward transformation.'
        )

    @classmethod
    def identity(cls, dimensionality=None):
        return cls(params=None, dimensionality=dimensionality)