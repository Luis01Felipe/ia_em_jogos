"""As IAs clássicas do Snake: ninguém aprende nada, tudo é regra ou busca escrita à mão."""

from collections import deque
from typing import Protocol

from arena.pathfinding.search import a_star_search, find_path, manhattan
from arena.snake.game import Direction, Point, SnakeGame


class SnakeAgent(Protocol):
    name: str

    def choose(self, game: SnakeGame) -> Direction: ...


def safe_directions(game: SnakeGame) -> list[Direction]:
    """Direções que não matam a cobra no próximo passo."""
    return [direction for direction in Direction if not game.is_deadly(game.next_head(direction))]


def direction_between(origin: Point, target: Point) -> Direction:
    delta = (target[0] - origin[0], target[1] - origin[1])
    return next(direction for direction in Direction if direction.delta == delta)


def reachable_area(game: SnakeGame, start: Point) -> int:
    """Quantas casas livres dá para alcançar a partir de ``start`` (sem atravessar o corpo)."""
    blocked = set(game.body)
    seen, queue = {start}, deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cell = (x + dx, y + dy)
            inside = 0 <= cell[0] < game.size and 0 <= cell[1] < game.size
            if inside and cell not in blocked and cell not in seen:
                seen.add(cell)
                queue.append(cell)
    return len(seen)


def roomiest_direction(game: SnakeGame) -> Direction:
    """Plano B: o lado seguro com mais espaço livre (ou seguir reto, se não houver saída)."""
    options = safe_directions(game)
    if not options:
        return game.direction
    return max(options, key=lambda d: reachable_area(game, game.next_head(d)))


class GreedyAgent:
    """Regras if/else: vai na direção que mais aproxima da maçã, desde que não morra agora."""

    name = "Regras if/else"

    def choose(self, game: SnakeGame) -> Direction:
        options = safe_directions(game)
        apple = game.apple
        if not options or apple is None:
            return game.direction
        return min(options, key=lambda d: manhattan(game.next_head(d), apple))


class AStarAgent:
    """Calcula com A* o caminho mais curto até a maçã, desviando do corpo."""

    name = "A* até a maçã"

    def choose(self, game: SnakeGame) -> Direction:
        if game.apple is None:
            return game.direction

        def free_neighbors(cell: Point) -> list[Point]:
            x, y = cell
            candidates = ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            return [c for c in candidates if not game.is_deadly(c)]

        path = find_path(a_star_search(game.head, game.apple, free_neighbors, manhattan))
        if len(path) >= 2:
            return direction_between(path[0], path[1])
        return roomiest_direction(game)


def hamiltonian_cycle(size: int) -> dict[Point, Point]:
    """Um circuito que passa por TODAS as casas uma vez e volta ao início (tabuleiro de lado par).

    Linha 0 da esquerda para a direita; depois zigue-zague pelas colunas 1..n-1 descendo;
    por fim sobe pela coluna 0 até o começo.
    """
    if size % 2:
        raise ValueError("O circuito simples precisa de um tabuleiro de lado par.")
    order = [(x, 0) for x in range(size)]
    for y in range(1, size):
        columns = range(size - 1, 0, -1) if y % 2 else range(1, size)
        order += [(x, y) for x in columns]
    order += [(0, y) for y in range(size - 1, 0, -1)]
    return {cell: order[(i + 1) % len(order)] for i, cell in enumerate(order)}


class HamiltonianAgent:
    """Ignora a maçã e percorre sempre o mesmo circuito por todas as casas: nunca bate."""

    name = "Ciclo hamiltoniano"

    def __init__(self, size: int) -> None:
        self._next_cell = hamiltonian_cycle(size)

    def choose(self, game: SnakeGame) -> Direction:
        return direction_between(game.head, self._next_cell[game.head])
