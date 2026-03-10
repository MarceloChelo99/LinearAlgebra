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


def test_execute_assignment_and_operation():
    viz = LinearAlgebraVisualizer()
    viz.execute("v=[1,2,3]")
    viz.execute("w=[3,2,1]")
    out = viz.execute("v+w")
    np.testing.assert_allclose(out.value, np.array([4.0, 4.0, 4.0]))


def test_dimensionality_reduction_shape():
    viz = LinearAlgebraVisualizer()
    reduced = viz.dimensionality_reduction([[1, 2, 3], [2, 3, 4], [3, 4, 5]], k=2)
    assert reduced.shape == (3, 2)


def test_main_starts_visualizer(monkeypatch):
    from la_visualizer import main as main_module

    started = {"called": False}

    def fake_run(self):
        started["called"] = True

    monkeypatch.setattr(main_module.LinearAlgebraVisualizer, "run_interactive", fake_run)
    main_module.main()

    assert started["called"] is True
