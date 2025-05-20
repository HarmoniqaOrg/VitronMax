import numpy as np


def test_featurise_numpy_shape() -> None:
    from app.predict import BBBPredictor, FP_BITS

    fp = BBBPredictor()._featurise("CCO")  # ethanol
    assert fp.shape == (FP_BITS,)
    assert fp.dtype == np.int8
    assert set(fp).issubset({0, 1})
