from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class OperationResult:
    value: float | np.ndarray
    summary: str


@dataclass
class SceneState:
    vectors: list[np.ndarray] = field(default_factory=list)
    points: list[np.ndarray] = field(default_factory=list)
    message: str = ""


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, symbols: dict[str, np.ndarray | float], functions: dict[str, Any]) -> None:
        self.symbols = symbols
        self.functions = functions

    def visit_Expression(self, node: ast.Expression):
        return self.visit(node.body)

    def visit_Name(self, node: ast.Name):
        if node.id in self.symbols:
            return self.symbols[node.id]
        raise ValueError(f"Unknown symbol '{node.id}'.")

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("Only numeric literals are allowed.")

    def visit_UnaryOp(self, node: ast.UnaryOp):
        value = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -value
        raise ValueError("Unsupported unary operation.")

    def visit_BinOp(self, node: ast.BinOp):
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return np.asarray(left) + np.asarray(right)
        if isinstance(node.op, ast.Sub):
            return np.asarray(left) - np.asarray(right)
        if isinstance(node.op, ast.Mult):
            if np.isscalar(left) or np.isscalar(right):
                return np.asarray(left) * np.asarray(right)
            raise ValueError("Use '@' for matrix multiplication.")
        if isinstance(node.op, ast.MatMult):
            return np.asarray(left) @ np.asarray(right)
        raise ValueError("Unsupported binary operation.")

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call.")
        name = node.func.id
        if name not in self.functions:
            raise ValueError(f"Unknown function '{name}'.")
        args = [self.visit(arg) for arg in node.args]
        return self.functions[name](*args)

    def generic_visit(self, node: ast.AST):
        raise ValueError(f"Unsupported expression: {type(node).__name__}")


class LinearAlgebraVisualizer:
    """Compute and interactively visualize vector and matrix operations with pygame."""

    def __init__(self) -> None:
        self.symbols: dict[str, np.ndarray | float] = {}
        self.scene = SceneState()

    def _to_vector(self, vector: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vector, dtype=float)
        if arr.ndim != 1 or arr.size not in (2, 3):
            raise ValueError("Vectors must be 2D or 3D.")
        return arr

    def dot_product(self, u: Sequence[float], v: Sequence[float]) -> float:
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")
        return float(np.dot(u_arr, v_arr))

    def projection(self, u: Sequence[float], v: Sequence[float]) -> np.ndarray:
        u_arr = self._to_vector(u)
        v_arr = self._to_vector(v)
        if u_arr.size != v_arr.size:
            raise ValueError("Vectors must have the same dimensionality.")
        denom = float(np.dot(v_arr, v_arr))
        if math.isclose(denom, 0.0):
            raise ValueError("Cannot project onto zero vector.")
        return (np.dot(u_arr, v_arr) / denom) * v_arr

    def dimensionality_reduction(self, data: Sequence[Sequence[float]], k: int = 2) -> np.ndarray:
        x = np.asarray(data, dtype=float)
        if x.ndim != 2:
            raise ValueError("Data must be a 2D matrix of samples.")
        if k < 1 or k > min(x.shape):
            raise ValueError("k must be between 1 and min(samples, features).")
        centered = x - x.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = vt[:k].T
        return centered @ components

    def _literal_or_symbol(self, text: str) -> np.ndarray | float:
        text = text.strip()
        if text in self.symbols:
            return self.symbols[text]
        value = ast.literal_eval(text)
        if isinstance(value, (int, float)):
            return float(value)
        arr = np.asarray(value, dtype=float)
        if arr.ndim not in (1, 2):
            raise ValueError("Only scalars, vectors, and matrices are supported.")
        return arr

    def evaluate(self, expression: str) -> np.ndarray | float:
        functions = {
            "proj": lambda u, v: self.projection(np.asarray(u), np.asarray(v)),
            "dot": lambda u, v: float(np.dot(np.asarray(u), np.asarray(v))),
            "dr": lambda data, k=2: self.dimensionality_reduction(np.asarray(data), int(k)),
            "pca": lambda data, k=2: self.dimensionality_reduction(np.asarray(data), int(k)),
        }
        tree = ast.parse(expression, mode="eval")
        evaluator = _SafeEvaluator(self.symbols, functions)
        return evaluator.visit(tree)

    def execute(self, command: str) -> OperationResult:
        text = command.strip()
        if not text:
            raise ValueError("Empty command.")

        if "=" in text and "==" not in text:
            name, expr = text.split("=", 1)
            var_name = name.strip()
            if not var_name.isidentifier():
                raise ValueError("Invalid variable name.")
            value = self.evaluate(expr.strip()) if any(op in expr for op in "+-@*()") or expr.strip().isidentifier() else self._literal_or_symbol(expr)
            arr = np.asarray(value) if not np.isscalar(value) else value
            self.symbols[var_name] = arr
            result = arr
            summary = f"Stored '{var_name}'"
        else:
            try:
                result = self.evaluate(text)
            except Exception:
                result = self._literal_or_symbol(text)
            summary = "Computed expression"

        self._update_scene(result)
        return OperationResult(value=result, summary=summary)

    def _update_scene(self, value: float | np.ndarray) -> None:
        self.scene = SceneState()
        if np.isscalar(value):
            self.scene.message = f"Scalar result: {float(value):.4g}"
            return

        arr = np.asarray(value, dtype=float)
        if arr.ndim == 1:
            self.scene.vectors = [arr]
            self.scene.message = f"Vector result shape={arr.shape}"
            return
        if arr.ndim == 2:
            if arr.shape[1] in (2, 3) and arr.shape[0] > 2:
                self.scene.points = [p for p in arr]
                self.scene.message = f"Point cloud shape={arr.shape}"
            else:
                self.scene.vectors = [arr[:, i] for i in range(arr.shape[1])]
                self.scene.message = f"Matrix result shape={arr.shape} (columns shown as vectors)"
            return
        raise ValueError("Cannot visualize result.")

    def run_interactive(self, width: int = 1200, height: int = 800) -> None:
        """Start a pygame app with command input, rotation, zoom, and reset controls."""
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Linear Algebra Visualizer (pygame)")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("consolas", 22)

        input_text = ""
        logs = ["Enter expressions like: v=[2,1,3], w=[1,-1,2], v+w, dot(v,w), A@v, dr(data,2)"]
        yaw, pitch, zoom = 0.8, -0.4, 140.0
        dragging = False

        def to_3d(vec: np.ndarray) -> np.ndarray:
            if vec.size == 2:
                return np.array([vec[0], vec[1], 0.0])
            if vec.size == 3:
                return vec
            raise ValueError("Only 2D/3D vectors are supported for rendering.")

        def project(point: np.ndarray) -> tuple[int, int]:
            nonlocal yaw, pitch, zoom
            cy, sy = math.cos(yaw), math.sin(yaw)
            cp, sp = math.cos(pitch), math.sin(pitch)
            rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            rot_x = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
            p = rot_x @ (rot_y @ point)
            cx, cy2 = width // 2, int(height * 0.42)
            return int(cx + p[0] * zoom), int(cy2 - p[1] * zoom)

        def draw_arrow(start: np.ndarray, end: np.ndarray, color: tuple[int, int, int]) -> None:
            s = project(start)
            e = project(end)
            pygame.draw.line(screen, color, s, e, 3)
            direction = np.array([e[0] - s[0], e[1] - s[1]], dtype=float)
            norm = np.linalg.norm(direction)
            if norm > 0:
                u = direction / norm
                left = np.array([-u[1], u[0]])
                p1 = np.array(e) - 14 * u + 7 * left
                p2 = np.array(e) - 14 * u - 7 * left
                pygame.draw.polygon(screen, color, [e, p1.astype(int), p2.astype(int)])

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                    elif event.button == 4:
                        zoom *= 1.1
                    elif event.button == 5:
                        zoom /= 1.1
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    dragging = False
                elif event.type == pygame.MOUSEMOTION and dragging:
                    dx, dy = event.rel
                    yaw += dx * 0.01
                    pitch += dy * 0.01
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        if input_text.strip().lower() == "reset":
                            self.scene = SceneState()
                            logs.append("Scene reset.")
                        else:
                            try:
                                out = self.execute(input_text)
                                logs.append(f"> {input_text}")
                                logs.append(out.summary)
                                logs.append(self.scene.message)
                            except Exception as exc:
                                logs.append(f"Error: {exc}")
                        input_text = ""
                    elif event.key == pygame.K_BACKSPACE:
                        input_text = input_text[:-1]
                    elif event.key == pygame.K_r:
                        yaw, pitch, zoom = 0.8, -0.4, 140.0
                        logs.append("View reset.")
                    elif event.key == pygame.K_c:
                        self.scene = SceneState()
                        logs.append("Cleared visualization.")
                    else:
                        if event.unicode and event.key != pygame.K_TAB:
                            input_text += event.unicode

            screen.fill((18, 18, 26))

            origin = np.zeros(3)
            axis_len = 2.0
            draw_arrow(origin, np.array([axis_len, 0, 0]), (220, 80, 80))
            draw_arrow(origin, np.array([0, axis_len, 0]), (80, 220, 80))
            draw_arrow(origin, np.array([0, 0, axis_len]), (80, 120, 230))

            for vec in self.scene.vectors:
                draw_arrow(origin, to_3d(np.asarray(vec, dtype=float)), (240, 200, 90))

            for point in self.scene.points:
                p = project(to_3d(np.asarray(point, dtype=float)))
                pygame.draw.circle(screen, (170, 230, 255), p, 4)

            pygame.draw.rect(screen, (28, 28, 38), (20, height - 64, width - 40, 40), border_radius=5)
            screen.blit(font.render(input_text or "Type expression and press Enter", True, (230, 230, 230)), (32, height - 56))

            panel_y = int(height * 0.72)
            pygame.draw.rect(screen, (24, 24, 34), (20, panel_y, width - 40, int(height * 0.23)), border_radius=5)
            for i, line in enumerate(logs[-7:]):
                screen.blit(font.render(line, True, (210, 210, 210)), (30, panel_y + 10 + i * 26))

            helper = "Drag mouse to rotate | Wheel to zoom | R reset view | C clear scene | command: reset"
            screen.blit(font.render(helper, True, (180, 180, 180)), (20, 20))

            pygame.display.flip()
            clock.tick(60)

        pygame.quit()
