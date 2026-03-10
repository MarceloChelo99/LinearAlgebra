from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class _ProjectionResult:
    scalar_component: float
    projected_vector: np.ndarray


class LinearAlgebraVisualizer:
    """Visualize vectors and matrix operations in up to 3 dimensions."""

    def _to_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.ndim != 1:
            raise ValueError("A vector must be one-dimensional.")
        if arr.size not in (2, 3):
            raise ValueError("Only 2D and 3D vectors are supported.")
        return arr

    def _to_vectors(self, vectors: Iterable[Sequence[float]]) -> list[np.ndarray]:
        parsed = [self._to_vector(v) for v in vectors]
        if not parsed:
            raise ValueError("At least one vector is required.")
        dims = {v.size for v in parsed}
        if len(dims) != 1:
            raise ValueError("All vectors must have the same dimensionality.")
        return parsed

    def _setup_axes(self, dim: int, title: str | None = None):
        fig = plt.figure(figsize=(8, 6))
        if dim == 2:
            ax = fig.add_subplot(111)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        else:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

        if title:
            ax.set_title(title)
        ax.grid(True)
        return fig, ax

    def _set_equal_axes(self, ax, vectors: list[np.ndarray]) -> None:
        max_abs = max(np.max(np.abs(v)) for v in vectors)
        limit = max(1.0, float(max_abs) * 1.3)

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        if vectors[0].size == 3:
            ax.set_zlim(-limit, limit)

    def _compute_projection(self, u: np.ndarray, v: np.ndarray) -> _ProjectionResult:
        v_norm_sq = float(np.dot(v, v))
        if np.isclose(v_norm_sq, 0.0):
            raise ValueError("Cannot project onto the zero vector.")
        scalar = float(np.dot(u, v) / v_norm_sq)
        projection = scalar * v
        return _ProjectionResult(scalar_component=scalar, projected_vector=projection)

    def dot_product(self, u: Sequence[float], v: Sequence[float]) -> float:
        """Return the dot product u·v."""
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")
        return float(np.dot(u_arr, v_arr))

    def projection(self, u: Sequence[float], v: Sequence[float]) -> np.ndarray:
        """Return projection of u onto v."""
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")
        return self._compute_projection(u_arr, v_arr).projected_vector

    def plot_vectors(
        self,
        vectors: Iterable[Sequence[float]],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
        title: str | None = None,
        show: bool = True,
    ):
        """Plot one or more vectors from the origin in 2D or 3D."""
        parsed = self._to_vectors(vectors)
        dim = parsed[0].size
        _, ax = self._setup_axes(dim, title or "Vector Visualization")

        if labels and len(labels) != len(parsed):
            raise ValueError("labels must match number of vectors.")
        if colors and len(colors) != len(parsed):
            raise ValueError("colors must match number of vectors.")

        for i, vec in enumerate(parsed):
            label = labels[i] if labels else f"v{i+1}"
            color = colors[i] if colors else None
            if dim == 2:
                ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color=color)
                ax.text(vec[0], vec[1], f" {label}")
            else:
                ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=color)
                ax.text(vec[0], vec[1], vec[2], f" {label}")

        self._set_equal_axes(ax, parsed)
        if show:
            plt.show()
        return ax

    def plot_span(
        self,
        u: Sequence[float],
        v: Sequence[float],
        title: str | None = None,
        show: bool = True,
    ):
        """Plot span(u, v) in 2D or 3D."""
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")

        dim = u_arr.size
        _, ax = self._setup_axes(dim, title or "Span Visualization")

        # Draw basis vectors
        if dim == 2:
            ax.quiver(0, 0, u_arr[0], u_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:blue")
            ax.quiver(0, 0, v_arr[0], v_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:orange")

            ts = np.linspace(-2, 2, 21)
            ss = np.linspace(-2, 2, 21)
            points = np.array([t * u_arr + s * v_arr for t in ts for s in ss])
            ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.2, color="gray")
        else:
            ax.quiver(0, 0, 0, u_arr[0], u_arr[1], u_arr[2], color="tab:blue")
            ax.quiver(0, 0, 0, v_arr[0], v_arr[1], v_arr[2], color="tab:orange")

            ts = np.linspace(-1.5, 1.5, 15)
            ss = np.linspace(-1.5, 1.5, 15)
            points = np.array([t * u_arr + s * v_arr for t in ts for s in ss])
            ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], alpha=0.25, color="gray", linewidth=0)

        self._set_equal_axes(ax, [u_arr, v_arr])
        if show:
            plt.show()
        return ax

    def plot_dot_product(
        self,
        u: Sequence[float],
        v: Sequence[float],
        title: str | None = None,
        show: bool = True,
    ):
        """Plot vectors and geometric projection relationship for dot product."""
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")

        dim = u_arr.size
        dot = float(np.dot(u_arr, v_arr))
        proj = self._compute_projection(u_arr, v_arr)

        _, ax = self._setup_axes(dim, title or f"Dot Product: u·v = {dot:.3g}")

        if dim == 2:
            ax.quiver(0, 0, u_arr[0], u_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:blue")
            ax.quiver(0, 0, v_arr[0], v_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:orange")
            p = proj.projected_vector
            ax.quiver(0, 0, p[0], p[1], angles="xy", scale_units="xy", scale=1, color="tab:green")
            ax.plot([u_arr[0], p[0]], [u_arr[1], p[1]], linestyle="--", color="black", alpha=0.6)
        else:
            ax.quiver(0, 0, 0, u_arr[0], u_arr[1], u_arr[2], color="tab:blue")
            ax.quiver(0, 0, 0, v_arr[0], v_arr[1], v_arr[2], color="tab:orange")
            p = proj.projected_vector
            ax.quiver(0, 0, 0, p[0], p[1], p[2], color="tab:green")
            ax.plot([u_arr[0], p[0]], [u_arr[1], p[1]], [u_arr[2], p[2]], linestyle="--", color="black", alpha=0.6)

        self._set_equal_axes(ax, [u_arr, v_arr, p])
        if show:
            plt.show()
        return ax

    def plot_projection(
        self,
        u: Sequence[float],
        v: Sequence[float],
        title: str | None = None,
        show: bool = True,
    ):
        """Plot vector u, target vector v, and projection of u onto v."""
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")

        proj = self._compute_projection(u_arr, v_arr).projected_vector
        dim = u_arr.size
        _, ax = self._setup_axes(dim, title or "Vector Projection")

        if dim == 2:
            ax.quiver(0, 0, u_arr[0], u_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:blue")
            ax.quiver(0, 0, v_arr[0], v_arr[1], angles="xy", scale_units="xy", scale=1, color="tab:orange")
            ax.quiver(0, 0, proj[0], proj[1], angles="xy", scale_units="xy", scale=1, color="tab:green")
            ax.plot([u_arr[0], proj[0]], [u_arr[1], proj[1]], "k--", alpha=0.6)
        else:
            ax.quiver(0, 0, 0, u_arr[0], u_arr[1], u_arr[2], color="tab:blue")
            ax.quiver(0, 0, 0, v_arr[0], v_arr[1], v_arr[2], color="tab:orange")
            ax.quiver(0, 0, 0, proj[0], proj[1], proj[2], color="tab:green")
            ax.plot([u_arr[0], proj[0]], [u_arr[1], proj[1]], [u_arr[2], proj[2]], "k--", alpha=0.6)

        self._set_equal_axes(ax, [u_arr, v_arr, proj])
        if show:
            plt.show()
        return ax

    def plot_matrix_transform(
        self,
        matrix: Sequence[Sequence[float]],
        vectors: Iterable[Sequence[float]],
        title: str | None = None,
        show: bool = True,
    ):
        """Plot original vectors and transformed vectors under matrix multiplication."""
        parsed = self._to_vectors(vectors)
        dim = parsed[0].size

        mat = np.asarray(matrix, dtype=float)
        if mat.shape != (dim, dim):
            raise ValueError(f"matrix must have shape {(dim, dim)}")

        transformed = [mat @ vec for vec in parsed]

        _, ax = self._setup_axes(dim, title or "Matrix Transformation")

        for i, vec in enumerate(parsed):
            t_vec = transformed[i]
            if dim == 2:
                ax.quiver(0, 0, vec[0], vec[1], angles="xy", scale_units="xy", scale=1, color="tab:blue", alpha=0.55)
                ax.quiver(0, 0, t_vec[0], t_vec[1], angles="xy", scale_units="xy", scale=1, color="tab:red", alpha=0.75)
                ax.text(vec[0], vec[1], f" v{i+1}")
                ax.text(t_vec[0], t_vec[1], f" T(v{i+1})")
            else:
                ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color="tab:blue", alpha=0.55)
                ax.quiver(0, 0, 0, t_vec[0], t_vec[1], t_vec[2], color="tab:red", alpha=0.75)
                ax.text(vec[0], vec[1], vec[2], f" v{i+1}")
                ax.text(t_vec[0], t_vec[1], t_vec[2], f" T(v{i+1})")

        self._set_equal_axes(ax, parsed + transformed)
        if show:
            plt.show()
        return ax
