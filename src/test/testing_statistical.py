import numpy as np
import statistical as mystat


def test_euclidean_distance():
    """Test the euclidean distance function if it gets the values correct"""
    testArray = np.array([[2, 2, 2], [1, 1, 1], [3, 3, 3], [1, 1, 1]])

    assert np.all(
        mystat.euclidean_norm(testArray, [0, 2], [1, 3]) == np.array([np.sqrt(3), 0])
    )
