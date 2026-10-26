"""Jogadores que podem sentar à mesa: um que joga ao acaso e um que calcula (Minimax)."""

import random
from dataclasses import dataclass, field
from typing import Protocol

from arena.minimax.games import TwoPlayerGame
from arena.minimax.search import choose_move


class Player(Protocol):
    name: str

    def choose(self, game: TwoPlayerGame) -> int: ...


@dataclass
class RandomPlayer:
    rng: random.Random
    name: str = "Aleatório"

    def choose(self, game: TwoPlayerGame) -> int:
        return self.rng.choice(game.legal_moves())


@dataclass
class MinimaxPlayer:
    depth: int
    name: str = "Minimax"
    prune: bool = True
    positions_examined: int = field(default=0, init=False)

    def choose(self, game: TwoPlayerGame) -> int:
        decision = choose_move(game, self.depth, prune=self.prune)
        self.positions_examined += decision.positions_examined
        return decision.move
