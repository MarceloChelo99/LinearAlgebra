# Linear Algebra Visualizer

A small Python module for visualizing core linear algebra ideas in 2D and 3D using `matplotlib`.

## Features

- Visualize one or more vectors in 2D or 3D.
- Visualize the span of two vectors (line in dependent case, plane in independent case).
- Visualize dot products geometrically (including the projection component).
- Visualize vector projection onto another vector.
- Visualize matrix transformations in 2D and 3D.

## Installation

```bash
pip install numpy matplotlib
```

## Quick start

```python
from la_visualizer import LinearAlgebraVisualizer

viz = LinearAlgebraVisualizer()

# Plot vectors
viz.plot_vectors([[2, 1], [1, 3]], labels=["u", "v"], title="2D Vectors")

# Plot span
viz.plot_span([2, 1], [1, 3], title="Span(u, v)")

# Dot product visualization
viz.plot_dot_product([3, 1], [2, 2], title="Dot Product")

# Projection visualization
viz.plot_projection([3, 1], [2, 2], title="Projection of u onto v")

# Matrix transform visualization
viz.plot_matrix_transform(
    matrix=[[1, -1], [1, 1]],
    vectors=[[1, 0], [0, 1], [1, 1]],
    title="2D Matrix Transform"
)
```

Use `show=False` on plotting methods if you want to compose multiple plots and call `matplotlib.pyplot.show()` manually.
