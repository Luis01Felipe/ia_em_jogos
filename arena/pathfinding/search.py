"""Busca em largura (BFS) e A*, escritas como geradores.

Cada ``yield`` entrega a casa que acabou de ser examinada — é isso que o vídeo pinta de
azul. Quando a busca termina, o gerador devolve (``return``) o caminho encontrado.
Use ``find_path`` quando só interessa o caminho final.
"""

import heapq
from collections import deque
from collections.abc import Callable, Generator, Iterable
from itertools import count

Cell = tuple[int, int]
Neighbors = Callable[[Cell], Iterable[Cell]]
Heuristic = Callable[[Cell, Cell], float]
Search = Generator[Cell, None, list[Cell]]


def manhattan(a: Cell, b: Cell) -> float:
    """Distância em "quarteirões": nunca superestima quando só dá para andar em 4 direções."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def breadth_first_search(start: Cell, goal: Cell, neighbors: Neighbors) -> Search:
    """Busca cega: examina as casas em ondas, da mais próxima da largada para a mais distante."""
    came_from: dict[Cell, Cell | None] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        yield cell
        if cell == goal:
            return _rebuild_path(came_from, goal)
        for neighbor in neighbors(cell):
            if neighbor not in came_from:
                came_from[neighbor] = cell
                queue.append(neighbor)
    return []


def a_star_search(start: Cell, goal: Cell, neighbors: Neighbors, heuristic: Heuristic = manhattan) -> Search:
    """Examina primeiro a casa com menor f = g (quanto já andei) + h (palpite de quanto falta)."""
    tie_breaker = count()
    cost_so_far: dict[Cell, float] = {start: 0}
    came_from: dict[Cell, Cell | None] = {start: None}
    # Em empates de f, prefere quem tem o menor h (está mais perto do objetivo).
    frontier = [(heuristic(start, goal), heuristic(start, goal), next(tie_breaker), start)]
    examined: set[Cell] = set()
    while frontier:
        *_, cell = heapq.heappop(frontier)
        if cell in examined:
            continue
        examined.add(cell)
        yield cell
        if cell == goal:
            return _rebuild_path(came_from, goal)
        for neighbor in neighbors(cell):
            new_cost = cost_so_far[cell] + 1
            if new_cost >= cost_so_far.get(neighbor, float("inf")):
                continue
            cost_so_far[neighbor] = new_cost
            came_from[neighbor] = cell
            h = heuristic(neighbor, goal)
            heapq.heappush(frontier, (new_cost + h, h, next(tie_breaker), neighbor))
    return []


def find_path(search: Search) -> list[Cell]:
    """Roda a busca até o fim e devolve só o caminho (vazio se não houver)."""
    while True:
        try:
            next(search)
        except StopIteration as finished:
            return finished.value


def _rebuild_path(came_from: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
    path = [goal]
    while (previous := came_from[path[-1]]) is not None:
        path.append(previous)
    return path[::-1]
