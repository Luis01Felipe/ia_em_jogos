"""Minimax com poda alfa-beta opcional e contador de posições examinadas.

MAX é quem está escolhendo a jogada (a IA); MIN é o adversário, que supomos perfeito.
"""

import math
from dataclasses import dataclass

from arena.minimax.games import TwoPlayerGame

WIN_SCORE = 10_000.0


@dataclass(frozen=True, slots=True)
class Decision:
    move: int
    score: float
    positions_examined: int
    depth: int

    def moves_until_forced_win(self) -> int | None:
        """Se a IA já enxerga uma vitória que o adversário não consegue impedir, em quantos lances."""
        if self.score < WIN_SCORE:
            return None
        remaining_depth = round(self.score - WIN_SCORE)
        plies = self.depth - remaining_depth
        return (plies + 1) // 2


class _Counter:
    def __init__(self) -> None:
        self.value = 0


def choose_move(game: TwoPlayerGame, depth: int, *, prune: bool = True) -> Decision:
    """Escolhe a melhor jogada olhando ``depth`` lances (meus e do adversário) à frente."""
    moves = game.legal_moves()
    if not moves:
        raise ValueError("Não há jogadas possíveis: o jogo acabou.")

    me = game.current_player
    counter = _Counter()
    best_move, best_score = moves[0], -math.inf
    alpha, beta = -math.inf, math.inf
    for move in moves:
        score = _minimax(game.play(move), depth - 1, alpha, beta, me, counter, prune)
        if score > best_score:
            best_move, best_score = move, score
        if prune:
            alpha = max(alpha, best_score)
    return Decision(best_move, best_score, counter.value, depth)


def _minimax(
    state: TwoPlayerGame,
    depth: int,
    alpha: float,
    beta: float,
    me: int,
    counter: _Counter,
    prune: bool,
) -> float:
    counter.value += 1

    winner = state.winner()
    if winner is not None:
        # Somar a profundidade restante faz a IA preferir vencer cedo e perder tarde.
        return WIN_SCORE + depth if winner == me else -(WIN_SCORE + depth)
    moves = state.legal_moves()
    if not moves:
        return 0.0  # empate
    if depth == 0:
        return state.heuristic(me)

    if state.current_player == me:
        value = -math.inf
        for move in moves:
            value = max(value, _minimax(state.play(move), depth - 1, alpha, beta, me, counter, prune))
            alpha = max(alpha, value)
            if prune and alpha >= beta:
                break  # MIN já tem uma resposta melhor em outro ramo: este não muda nada
        return value

    value = math.inf
    for move in moves:
        value = min(value, _minimax(state.play(move), depth - 1, alpha, beta, me, counter, prune))
        beta = min(beta, value)
        if prune and alpha >= beta:
            break  # MAX já tem uma jogada melhor em outro ramo: este não muda nada
    return value


@dataclass(frozen=True, slots=True)
class PruningComparison:
    """A mesma decisão calculada duas vezes: sem poda e com poda alfa-beta."""

    with_pruning: Decision
    without_pruning: Decision

    @property
    def savings(self) -> float:
        """Fração de posições que a poda deixou de examinar (0.97 = 97% a menos)."""
        return 1 - self.with_pruning.positions_examined / self.without_pruning.positions_examined


def compare_pruning(game: TwoPlayerGame, depth: int) -> PruningComparison:
    with_pruning = choose_move(game, depth, prune=True)
    without_pruning = choose_move(game, depth, prune=False)
    if with_pruning.move != without_pruning.move:
        raise AssertionError("A poda nunca deveria mudar a jogada escolhida.")
    return PruningComparison(with_pruning, without_pruning)
