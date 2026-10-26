"""Mapas em grade: um labirinto com vários caminhos e um campo aberto com obstáculos."""

import random
from dataclasses import dataclass

from arena.pathfinding.search import Cell

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@dataclass(frozen=True)
class GridMap:
    width: int
    height: int
    walls: frozenset[Cell]
    start: Cell
    goal: Cell

    def is_open(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.walls

    def neighbors(self, cell: Cell) -> list[Cell]:
        x, y = cell
        return [(x + dx, y + dy) for dx, dy in DIRECTIONS if self.is_open((x + dx, y + dy))]


def maze_with_loops(width: int, height: int, rng: random.Random, extra_openings: float = 0.08) -> GridMap:
    """Labirinto gerado por backtracking; depois abrimos algumas paredes para criar atalhos.

    ``width`` e ``height`` precisam ser ímpares (corredores nas casas ímpares).
    """
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("Use largura e altura ímpares.")
    walls = {(x, y) for x in range(width) for y in range(height)}
    start = (1, 1)
    walls.discard(start)
    stack = [start]
    while stack:
        x, y = stack[-1]
        options = [
            (x + 2 * dx, y + 2 * dy, dx, dy)
            for dx, dy in DIRECTIONS
            if (x + 2 * dx, y + 2 * dy) in walls and 0 < x + 2 * dx < width - 1 and 0 < y + 2 * dy < height - 1
        ]
        if not options:
            stack.pop()
            continue
        nx, ny, dx, dy = rng.choice(options)
        walls.discard((x + dx, y + dy))
        walls.discard((nx, ny))
        stack.append((nx, ny))

    inner_walls = [cell for cell in walls if 0 < cell[0] < width - 1 and 0 < cell[1] < height - 1]
    for cell in rng.sample(inner_walls, round(len(inner_walls) * extra_openings)):
        if _separates_two_corridors(cell, walls):
            walls.discard(cell)
    return GridMap(width, height, frozenset(walls), start, (width - 2, height - 2))


def _separates_two_corridors(cell: Cell, walls: set[Cell]) -> bool:
    x, y = cell
    horizontal = (x - 1, y) not in walls and (x + 1, y) not in walls
    vertical = (x, y - 1) not in walls and (x, y + 1) not in walls
    return horizontal != vertical


def open_field(width: int, height: int, rng: random.Random, density: float = 0.28) -> GridMap:
    """Campo aberto com obstáculos espalhados ao acaso (largada e chegada sempre livres)."""
    start, goal = (1, height // 2), (width - 2, height // 2)
    walls = {
        (x, y) for x in range(width) for y in range(height) if rng.random() < density and (x, y) not in (start, goal)
    }
    return GridMap(width, height, frozenset(walls), start, goal)
