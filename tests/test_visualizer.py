import numpy as np

from la_visualizer import LinearAlgebraVisualizer


def test_dot_product():
    viz = LinearAlgebraVisualizer()
    assert viz.dot_product([1, 2, 3], [4, 5, 6]) == 32.0


def test_projection():
    viz = LinearAlgebraVisualizer()
    proj = viz.projection([3, 1], [2, 2])
    np.testing.assert_allclose(proj, np.array([2.0, 2.0]))


def test_dimension_mismatch_raises():
    viz = LinearAlgebraVisualizer()
    try:
        viz.dot_product([1, 2], [1, 2, 3])
        assert False, "Expected ValueError"
    except ValueError:
        assert True
