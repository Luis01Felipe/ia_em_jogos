"""Contrato comum dos jogos de dois jogadores, por turnos e com informação perfeita.

O algoritmo Minimax (``arena.minimax.search``) só conhece este contrato: ele funciona
igualzinho no Jogo da Velha e no Ligue 4 porque os dois cumprem o mesmo Protocol.
"""

from typing import Protocol, Self

EMPTY = 0
FIRST = 1
SECOND = 2


def opponent(player: int) -> int:
    return SECOND if player == FIRST else FIRST


class TwoPlayerGame(Protocol):
    @property
    def current_player(self) -> int: ...

    def legal_moves(self) -> list[int]:
        """Jogadas possíveis, já na ordem em que vale a pena examiná-las."""
        ...

    def play(self, move: int) -> Self:
        """Devolve um NOVO estado com a jogada feita (o estado atual não muda)."""
        ...

    def winner(self) -> int | None: ...

    def heuristic(self, player: int) -> float:
        """Palpite de quão boa é a posição para ``player`` quando não dá para calcular até o fim."""
        ...
