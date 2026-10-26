"""Regras do Jogo da Velha. Tabuleiro: índices 0..8, linha por linha."""

from dataclasses import dataclass

from arena.minimax.games import EMPTY, FIRST, opponent

ROWS = ((0, 1, 2), (3, 4, 5), (6, 7, 8))
COLUMNS = ((0, 3, 6), (1, 4, 7), (2, 5, 8))
DIAGONALS = ((0, 4, 8), (2, 4, 6))
LINES = ROWS + COLUMNS + DIAGONALS


@dataclass(frozen=True, slots=True)
class TicTacToe:
    cells: tuple[int, ...] = (EMPTY,) * 9
    current_player: int = FIRST

    def legal_moves(self) -> list[int]:
        if self.winner() is not None:
            return []
        return [index for index, cell in enumerate(self.cells) if cell == EMPTY]

    def play(self, move: int) -> "TicTacToe":
        if self.cells[move] != EMPTY:
            raise ValueError(f"A casa {move} já está ocupada.")
        cells = list(self.cells)
        cells[move] = self.current_player
        return TicTacToe(tuple(cells), opponent(self.current_player))

    def winner(self) -> int | None:
        for a, b, c in LINES:
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return self.cells[a]
        return None

    def winning_line(self) -> tuple[int, int, int] | None:
        for line in LINES:
            a, b, c = line
            if self.cells[a] != EMPTY and self.cells[a] == self.cells[b] == self.cells[c]:
                return line
        return None

    def is_over(self) -> bool:
        return self.winner() is not None or EMPTY not in self.cells

    def heuristic(self, player: int) -> float:
        # O Jogo da Velha é pequeno: a busca sempre chega ao fim, nunca precisa de palpite.
        return 0.0
