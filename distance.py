import pyemd
from scipy import spatial
import numpy as np
import os
import joblib

os.makedirs('cache', exist_ok=True)
MEMORY = joblib.Memory(cachedir='cache', verbose=0)


def earth_movers_distance(distance_matrix, image1, image2):
    """Returns Earth Mover's Distance for image1 and image2.

    distance_matrix is an  N x N distance matrix where N = x * y * z
    where the shape of image1 and image2 are (x, y, z).
    distance_matrix[i][j] gives the distance between the ith and jth
    element of an unraveled image. See numpy.ravel() for details on
    how a three dimensional array is converted to a one dimensional
    array.
    """

    # turn voxel activations into probability distributions
    image1, image2 = [np.clip(img, 0, 999) for img in (image1, image2)]
    image1, image2 = [img / np.sum(img) for img in (image1, image2)]

    result = pyemd.emd(image1.ravel(), image2.ravel(), distance_matrix)
    return result


@MEMORY.cache
def euclidean_distance_matrix(shape):
    """Returns a distance matrix for all points in a space with given shape."""
    m = np.mgrid[:shape[0], :shape[1], :shape[2]]
    coords = np.array([m[0].ravel(), m[1].ravel(), m[2].ravel()]).T
    return spatial.distance.squareform(spatial.distance.pdist(coords))


def euclidean_emd(image1, image2):
    """Earth Movers Distance with a Euclidean distance matrix."""
    distance_matrix = euclidean_distance_matrix(image1.shape)
    return earth_movers_distance(distance_matrix, image1, image2)
