import random

import pytest

from arena.minimax.connect_four import ConnectFour
from arena.minimax.games import FIRST, SECOND
from arena.minimax.players import MinimaxPlayer, RandomPlayer
from arena.minimax.search import choose_move, compare_pruning
from arena.minimax.tic_tac_toe import TicTacToe


def play_out(game, players):
    while not game.is_over():
        game = game.play(players[game.current_player].choose(game))
    return game


def test_full_tic_tac_toe_tree_has_549_945_positions_below_the_root():
    decision = choose_move(TicTacToe(), depth=9, prune=False)
    assert decision.positions_examined == 549_945
    assert decision.score == 0  # jogo perfeito dos dois lados = empate


def test_pruning_examines_far_fewer_positions_and_keeps_the_move():
    comparison = compare_pruning(TicTacToe(), depth=9)
    assert comparison.with_pruning.move == comparison.without_pruning.move
    assert comparison.savings > 0.9


@pytest.mark.parametrize("seed", range(5))
def test_pruning_never_changes_the_connect_four_move(seed):
    rng = random.Random(seed)
    game = ConnectFour()
    for _ in range(rng.randint(0, 10)):
        game = game.play(rng.choice(game.legal_moves()))
    if game.is_over():
        return
    compare_pruning(game, depth=4)  # levanta AssertionError se a jogada mudar


def test_minimax_never_loses_tic_tac_toe_against_random():
    rng = random.Random(0)
    for game_number in range(40):
        minimax_symbol = FIRST if game_number % 2 == 0 else SECOND
        players = {minimax_symbol: MinimaxPlayer(depth=9), 3 - minimax_symbol: RandomPlayer(rng)}
        final = play_out(TicTacToe(), players)
        assert final.winner() in (minimax_symbol, None)


def test_minimax_against_itself_is_always_a_draw():
    final = play_out(TicTacToe(), {FIRST: MinimaxPlayer(depth=9), SECOND: MinimaxPlayer(depth=9)})
    assert final.winner() is None


def test_connect_four_takes_an_immediate_win():
    game = ConnectFour()
    for column in (0, 6, 0, 6, 0, 6):  # laranja empilha 3 na coluna 0
        game = game.play(column)
    decision = choose_move(game, depth=2)
    assert decision.move == 0
    assert decision.moves_until_forced_win() == 1


def test_connect_four_detects_diagonal_win():
    game = ConnectFour()
    for column in (0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3):
        game = game.play(column)
    assert game.winner() == FIRST
