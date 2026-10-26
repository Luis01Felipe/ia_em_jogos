"""Regras do Snake num tabuleiro quadrado. O jogo não sabe nada de recompensas nem de IA."""

import random
from collections import deque
from dataclasses import dataclass
from enum import IntEnum

Point = tuple[int, int]


class Direction(IntEnum):
    """Em sentido horário: virar à direita = +1, virar à esquerda = -1."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def delta(self) -> Point:
        return ((0, -1), (1, 0), (0, 1), (-1, 0))[self]

    def turned(self, steps: int) -> "Direction":
        return Direction((self + steps) % 4)


class Turn(IntEnum):
    """Ações relativas, do ponto de vista da cobra (usadas pelas IAs que aprendem)."""

    STRAIGHT = 0
    RIGHT = 1
    LEFT = 2

    def apply_to(self, direction: Direction) -> Direction:
        return direction.turned({Turn.STRAIGHT: 0, Turn.RIGHT: 1, Turn.LEFT: -1}[self])


@dataclass(frozen=True)
class StepOutcome:
    ate: bool
    died: bool


class SnakeGame:
    """``starvation_limit``: a cobra morre de fome se passar tantos passos sem comer (None = nunca)."""

    def __init__(self, size: int = 10, seed: int = 0, starvation_limit: int | None = 200) -> None:
        self.size = size
        self.starvation_limit = starvation_limit
        self._rng = random.Random(seed)
        # Começa na terceira linha, andando para a direita (comum a todas as IAs).
        self.body: deque[Point] = deque([(4, 2), (3, 2), (2, 2)])
        self.direction = Direction.RIGHT
        self.apple: Point | None = self._place_apple()
        self.score = 0
        self.steps = 0
        self.steps_since_apple = 0
        self.alive = True
        self.cause_of_death = ""

    @property
    def head(self) -> Point:
        return self.body[0]

    @property
    def won(self) -> bool:
        return self.apple is None

    @property
    def is_over(self) -> bool:
        return not self.alive or self.won

    def next_head(self, direction: Direction) -> Point:
        dx, dy = direction.delta
        return self.head[0] + dx, self.head[1] + dy

    def is_deadly(self, point: Point) -> bool:
        """Bater na parede ou no próprio corpo. A ponta do rabo sai do lugar, então não conta."""
        x, y = point
        if not (0 <= x < self.size and 0 <= y < self.size):
            return True
        growing = point == self.apple
        obstacles = list(self.body) if growing else list(self.body)[:-1]
        return point in obstacles

    def step(self, direction: Direction) -> StepOutcome:
        if self.is_over:
            raise RuntimeError("O jogo já acabou.")
        self.direction = direction
        new_head = self.next_head(direction)
        self.steps += 1
        self.steps_since_apple += 1

        if self.is_deadly(new_head):
            return self._die("bateu")
        self.body.appendleft(new_head)
        if new_head == self.apple:
            self.score += 1
            self.steps_since_apple = 0
            self.apple = self._place_apple()
            return StepOutcome(ate=True, died=False)
        self.body.pop()
        if self.starvation_limit is not None and self.steps_since_apple > self.starvation_limit:
            return self._die("fome")
        return StepOutcome(ate=False, died=False)

    def free_cells(self) -> list[Point]:
        occupied = set(self.body)
        return [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in occupied]

    def _die(self, cause: str) -> StepOutcome:
        self.alive = False
        self.cause_of_death = cause
        return StepOutcome(ate=False, died=True)

    def _place_apple(self) -> Point | None:
        free = self.free_cells()
        return self._rng.choice(free) if free else None
