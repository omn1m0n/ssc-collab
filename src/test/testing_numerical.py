import pandas as pd
import numpy as np
import numerical as mynum


def test_filter_constant():
    """Test if function correctly deletes columns"""

    testDf = pd.DataFrame(
        {"time": np.arange(5), "one": np.ones(5), "zero": np.zeros(5)}
    )
    resultDf = pd.DataFrame({"time": np.arange(5), "zero": np.zeros(5)})

    assert np.all(
        mynum.df_filter_constant(testDf, ["zero"], VarThreshold=0) == resultDf
    )


def test_autocorrelation():
    """test if the autocorrelation function is working as intended"""

    testDf = pd.DataFrame(
        {"one": [1, 0, 0], "two": [0, 1, 0], "three": [0, 0, 1], "four": [1, 1, 1]}
    )

    assert np.all(
        mynum.autocorrelation(testDf, ["four"])[-1]
        == np.array([1, 0, 0], dtype=complex)
    )
