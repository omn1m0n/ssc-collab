import numpy as np


def euclidean_norm(vectorList, listP, listQ):
    """Calculates the euclidean norm (distance) of two array-like objects, in this case vectors
    Args:
        listP (integer list): List of indices of the reference vector of the\
        array.
        list_comp (integer list): list of indices of vectors to compare to\
        data (numpy array): Data object with vectors to be analyzed.
    Returns:
        numpy array: The L2 norm (euclidean distance) between the P vectors and\
        Q vectors.
    """
    distanceArray = np.zeros(len(listP))
    for i in range(0, len(listP)):
        distanceArray[i] = np.linalg.norm(vectorList[listQ[i]] - vectorList[listP[i]])
    return distanceArray
