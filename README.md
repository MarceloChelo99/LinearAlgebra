# Linear Algebra Visualizer

An interactive linear algebra visualizer powered by `pygame` + `numpy`.

## What changed

This project now uses a real-time pygame canvas (instead of matplotlib) with:
- Command input for vectors, matrices, and operations.
- 3D-style graph rotation (mouse drag), zoom (mouse wheel), and view reset.
- Scene reset / clear for trying new operations quickly.
- Built-in dimensionality reduction (`dr(...)` / `pca(...)`) visualization support.

## Installation

```bash
pip install numpy pygame
```

## Run interactive tool

```python
from la_visualizer import LinearAlgebraVisualizer

viz = LinearAlgebraVisualizer()
viz.run_interactive()
```

## Command examples

- Variable assignment:
  - `v=[2,1,3]`
  - `w=[1,-1,2]`
  - `A=[[1,0,0],[0,0,-1],[0,1,0]]`
- Operations:
  - `v+w`
  - `A@v`
  - `dot(v,w)`
  - `proj(v,w)`
- Dimensionality reduction:
  - `data=[[2,1,0],[1,2,0],[3,4,1],[4,3,1]]`
  - `dr(data,2)`

## Controls

- **Enter**: execute command in input box.
- **Mouse drag**: rotate graph.
- **Mouse wheel**: zoom.
- **R**: reset camera view.
- **C**: clear current visualization.
- **`reset` command**: clear scene via input.
