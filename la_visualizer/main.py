from __future__ import annotations

from la_visualizer import LinearAlgebraVisualizer


def main() -> None:
    """Start the interactive visualizer window."""
    viz = LinearAlgebraVisualizer()
    viz.run_interactive()


if __name__ == "__main__":
    main()
