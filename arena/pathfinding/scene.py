"""Vídeo 2: busca em largura (BFS) e A* resolvendo o mesmo mapa, lado a lado.

Azul = casa examinada; laranja = caminho final. As duas buscas andam no mesmo ritmo
(mesmo número de casas examinadas por quadro), então quem termina antes examinou menos.
"""

import random
from dataclasses import dataclass, field

import pygame

from arena.pathfinding.grid import GridMap, maze_with_loops, open_field
from arena.pathfinding.search import Cell, Search, a_star_search, breadth_first_search
from arena.view import drawing, theme
from arena.view.output import Scene, hold

CELL = 12
MAP_WIDTH, MAP_HEIGHT = 47, 39
CELLS_PER_FRAME = 4
OPEN_COLOR = (40, 36, 54)
CURRENT_COLOR = (170, 228, 255)


@dataclass
class SearchRun:
    """Acompanha uma busca em andamento para poder desenhá-la quadro a quadro."""

    title: str
    search: Search
    examined: list[Cell] = field(default_factory=list)
    path: list[Cell] | None = None

    def advance(self, steps: int) -> None:
        for _ in range(steps):
            if self.path is not None:
                return
            try:
                self.examined.append(next(self.search))
            except StopIteration as finished:
                self.path = finished.value

    @property
    def finished(self) -> bool:
        return self.path is not None


def build_scene() -> Scene:
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    rounds = (
        ("Labirinto com vários caminhos", maze_with_loops(MAP_WIDTH, MAP_HEIGHT, random.Random(1))),
        ("Campo aberto com obstáculos", open_field(MAP_WIDTH, MAP_HEIGHT, random.Random(2))),
    )
    for number, (title, grid) in enumerate(rounds, start=1):
        yield from _race(surface, f"Mapa {number} de {len(rounds)}", title, grid)


def _race(surface: pygame.Surface, eyebrow: str, title: str, grid: GridMap) -> Scene:
    runs = (
        SearchRun("Busca em largura (BFS)", breadth_first_search(grid.start, grid.goal, grid.neighbors)),
        SearchRun("A*  (f = g + h)", a_star_search(grid.start, grid.goal, grid.neighbors)),
    )
    yield from hold(_draw(surface, eyebrow, title, grid, runs, path_progress=0), 1.0)
    while not all(run.finished for run in runs):
        for run in runs:
            run.advance(CELLS_PER_FRAME)
        yield _draw(surface, eyebrow, title, grid, runs, path_progress=0)
    for progress in range(1, 31):
        yield _draw(surface, eyebrow, title, grid, runs, path_progress=progress / 30)
    yield from hold(_draw(surface, eyebrow, title, grid, runs, path_progress=1, show_summary=True), 4.0)


def _draw(
    surface: pygame.Surface,
    eyebrow: str,
    title: str,
    grid: GridMap,
    runs: tuple[SearchRun, SearchRun],
    *,
    path_progress: float,
    show_summary: bool = False,
) -> pygame.Surface:
    surface.fill(theme.BACKGROUND)
    drawing.header(surface, eyebrow, title)
    for index, run in enumerate(runs):
        origin = (48 + index * 624, 150)
        _draw_run(surface, grid, run, origin, path_progress)
    if show_summary:
        _draw_summary(surface, runs)
    return surface


def _draw_run(
    surface: pygame.Surface, grid: GridMap, run: SearchRun, origin: tuple[int, int], path_progress: float
) -> None:
    x0, y0 = origin
    board = pygame.Rect(x0, y0, grid.width * CELL, grid.height * CELL)
    drawing.text(surface, run.title, (x0, y0 - 38), kind="title", size=24)
    pygame.draw.rect(surface, theme.WALL, board.inflate(8, 8), border_radius=8)

    examined = set(run.examined)
    for y in range(grid.height):
        for x in range(grid.width):
            color = theme.WALL if (x, y) in grid.walls else theme.EXPLORED if (x, y) in examined else OPEN_COLOR
            pygame.draw.rect(surface, color, (x0 + x * CELL, y0 + y * CELL, CELL, CELL))
    if run.examined and not run.finished:
        cx, cy = run.examined[-1]
        pygame.draw.rect(surface, CURRENT_COLOR, (x0 + cx * CELL, y0 + cy * CELL, CELL, CELL))

    path = run.path or []
    for x, y in path[: round(len(path) * path_progress)]:
        pygame.draw.rect(
            surface, theme.ORANGE, (x0 + x * CELL + 2, y0 + y * CELL + 2, CELL - 4, CELL - 4), border_radius=3
        )
    _draw_endpoint(surface, grid.start, origin, theme.BLUE, filled=True)
    _draw_endpoint(surface, grid.goal, origin, theme.ORANGE, filled=False)

    status_y = board.bottom + 16
    drawing.text(
        surface,
        f"casas examinadas: {drawing.thousands(len(run.examined))}",
        (x0, status_y),
        kind="mono_bold",
        size=20,
        color=theme.BLUE,
    )
    if run.finished:
        drawing.text(
            surface,
            f"caminho: {len(path) - 1} passos",
            (board.right, status_y),
            kind="mono_bold",
            size=20,
            color=theme.ORANGE,
            anchor="topright",
        )


def _draw_endpoint(
    surface: pygame.Surface, cell: Cell, origin: tuple[int, int], color: theme.Color, *, filled: bool
) -> None:
    center = (origin[0] + cell[0] * CELL + CELL // 2, origin[1] + cell[1] * CELL + CELL // 2)
    pygame.draw.circle(surface, color, center, CELL - 2, 0 if filled else 3)


def _draw_summary(surface: pygame.Surface, runs: tuple[SearchRun, SearchRun]) -> None:
    bfs, a_star = runs
    savings = 1 - len(a_star.examined) / len(bfs.examined)
    message = f"Mesmo caminho de {len(a_star.path or []) - 1} passos — o A* examinou {savings:.0%} menos casas"
    banner = pygame.Rect(48, 664, 1184, 44)
    pygame.draw.rect(surface, theme.TEXT, banner, border_radius=10)
    drawing.text(surface, message, banner.center, kind="title", size=24, color=theme.BACKGROUND, anchor="center")
