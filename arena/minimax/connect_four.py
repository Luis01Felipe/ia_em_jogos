"""Regras do Ligue 4 (6 linhas × 7 colunas) e a função de avaliação usada no limite da busca.

Índice de uma casa: ``linha * COLUMNS + coluna``, com a linha 0 embaixo.
"""

from dataclasses import dataclass

from arena.minimax.games import EMPTY, FIRST, opponent

ROWS = 6
COLUMNS = 7
CENTER_COLUMN = COLUMNS // 2

# Examinar primeiro as colunas do centro faz a poda alfa-beta cortar muito mais cedo.
COLUMN_ORDER = (3, 2, 4, 1, 5, 0, 6)


def cell_index(row: int, column: int) -> int:
    return row * COLUMNS + column


def _all_windows() -> tuple[tuple[int, ...], ...]:
    """Todas as 69 sequências de 4 casas onde alguém pode ligar 4."""
    directions = ((0, 1), (1, 0), (1, 1), (1, -1))
    windows = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            for d_row, d_column in directions:
                end_row, end_column = row + 3 * d_row, column + 3 * d_column
                if not (0 <= end_row < ROWS and 0 <= end_column < COLUMNS):
                    continue
                windows.append(tuple(cell_index(row + k * d_row, column + k * d_column) for k in range(4)))
    return tuple(windows)


WINDOWS = _all_windows()
WINDOWS_BY_CELL = tuple(tuple(window for window in WINDOWS if cell in window) for cell in range(ROWS * COLUMNS))


@dataclass(frozen=True, slots=True)
class ConnectFour:
    cells: tuple[int, ...] = (EMPTY,) * (ROWS * COLUMNS)
    current_player: int = FIRST
    last_cell: int | None = None

    def legal_moves(self) -> list[int]:
        if self.winner() is not None:
            return []
        top_row = ROWS - 1
        return [column for column in COLUMN_ORDER if self.cells[cell_index(top_row, column)] == EMPTY]

    def landing_row(self, column: int) -> int:
        for row in range(ROWS):
            if self.cells[cell_index(row, column)] == EMPTY:
                return row
        raise ValueError(f"A coluna {column} está cheia.")

    def play(self, move: int) -> "ConnectFour":
        cell = cell_index(self.landing_row(move), move)
        cells = list(self.cells)
        cells[cell] = self.current_player
        return ConnectFour(tuple(cells), opponent(self.current_player), cell)

    def winner(self) -> int | None:
        # Só a última peça pode ter criado uma sequência de 4: basta olhar as janelas dela.
        if self.last_cell is None:
            return None
        player = self.cells[self.last_cell]
        for window in WINDOWS_BY_CELL[self.last_cell]:
            if all(self.cells[cell] == player for cell in window):
                return player
        return None

    def winning_window(self) -> tuple[int, ...] | None:
        if self.winner() is None or self.last_cell is None:
            return None
        player = self.cells[self.last_cell]
        return next(w for w in WINDOWS_BY_CELL[self.last_cell] if all(self.cells[c] == player for c in w))

    def is_over(self) -> bool:
        return self.winner() is not None or EMPTY not in self.cells

    def heuristic(self, player: int) -> float:
        """Soma pontos por sequências promissoras; subtrai pelas do adversário."""
        rival = opponent(player)
        score = 0.0
        for row in range(ROWS):
            if self.cells[cell_index(row, CENTER_COLUMN)] == player:
                score += 3
        for window in WINDOWS:
            score += _window_score(window, self.cells, player, rival)
        return score


def _window_score(window: tuple[int, ...], cells: tuple[int, ...], player: int, rival: int) -> float:
    mine = theirs = empty = 0
    for cell in window:
        owner = cells[cell]
        if owner == player:
            mine += 1
        elif owner == rival:
            theirs += 1
        else:
            empty += 1
    if mine == 3 and empty == 1:
        return 5
    if mine == 2 and empty == 2:
        return 2
    if theirs == 3 and empty == 1:
        return -4
    return 0
