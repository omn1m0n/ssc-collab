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
