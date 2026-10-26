"""Vídeo 1: o Minimax no Jogo da Velha e no Ligue 4.

Parte 1 — Minimax contra um jogador aleatório e contra ele mesmo (Jogo da Velha).
Parte 2 — Ligue 4: quem olha 6 lances à frente contra quem olha só 2. A cada jogada
da IA mais profunda, a mesma decisão é calculada com e sem poda alfa-beta.
"""

import random
from collections.abc import Generator
from dataclasses import dataclass

import pygame

from arena.minimax.connect_four import COLUMNS, ROWS, ConnectFour, cell_index
from arena.minimax.games import EMPTY, FIRST, SECOND, opponent
from arena.minimax.players import MinimaxPlayer, Player, RandomPlayer
from arena.minimax.search import PruningComparison, choose_move, compare_pruning
from arena.minimax.tic_tac_toe import TicTacToe
from arena.view import drawing, theme
from arena.view.output import Scene, hold

SEED = 7
GAMES_AGAINST_RANDOM = 8
GAMES_AGAINST_ITSELF = 2
DEEP_SEARCH = 6
SHALLOW_SEARCH = 2

PLAYER_COLORS = {FIRST: theme.ORANGE, SECOND: theme.BLUE}


# ---------------------------------------------------------------- Parte 1: Jogo da Velha


@dataclass
class Scoreboard:
    wins: int = 0
    draws: int = 0
    losses: int = 0
    self_play_draws: int = 0
    self_play_games: int = 0


def tic_tac_toe_part(surface: pygame.Surface) -> Scene:
    rng = random.Random(SEED)
    minimax = MinimaxPlayer(depth=9)
    score = Scoreboard()

    for game_number in range(GAMES_AGAINST_RANDOM):
        minimax_symbol = FIRST if game_number % 2 == 0 else SECOND
        players = _seat(minimax, RandomPlayer(rng), minimax_symbol)
        final = yield from _play_tic_tac_toe(surface, players, score, "Minimax × jogador aleatório")
        winner = final.winner()
        if winner is None:
            score.draws += 1
        elif winner == minimax_symbol:
            score.wins += 1
        else:
            score.losses += 1
        yield from _tic_tac_toe_frames(surface, final, score, "Minimax × jogador aleatório", 14)

    for _ in range(GAMES_AGAINST_ITSELF):
        players = {FIRST: MinimaxPlayer(depth=9), SECOND: MinimaxPlayer(depth=9)}
        final = yield from _play_tic_tac_toe(surface, players, score, "Minimax × Minimax")
        score.self_play_games += 1
        if final.winner() is None:
            score.self_play_draws += 1
        yield from _tic_tac_toe_frames(surface, final, score, "Minimax × Minimax", 14)

    yield from _tic_tac_toe_frames(surface, final, score, "Minimax × Minimax", 45)


def _seat(minimax: Player, other: Player, minimax_symbol: int) -> dict[int, Player]:
    return {minimax_symbol: minimax, opponent(minimax_symbol): other}


def _play_tic_tac_toe(
    surface: pygame.Surface, players: dict[int, Player], score: Scoreboard, title: str
) -> Generator[pygame.Surface, None, TicTacToe]:
    game = TicTacToe()
    yield from _tic_tac_toe_frames(surface, game, score, title, 6)
    while not game.is_over():
        game = game.play(players[game.current_player].choose(game))
        yield from _tic_tac_toe_frames(surface, game, score, title, 7)
    return game


def _tic_tac_toe_frames(surface: pygame.Surface, game: TicTacToe, score: Scoreboard, title: str, frames: int) -> Scene:
    surface.fill(theme.BACKGROUND)
    drawing.header(surface, "Parte 1 · Jogo da Velha", title)
    _draw_tic_tac_toe(surface, game, pygame.Rect(110, 170, 450, 450))
    _draw_scoreboard(surface, score)
    yield from hold(surface, frames / theme.FPS)


def _draw_tic_tac_toe(surface: pygame.Surface, game: TicTacToe, area: pygame.Rect) -> None:
    drawing.panel(surface, area)
    cell = area.width // 3
    for k in (1, 2):
        pygame.draw.line(
            surface, theme.LINE, (area.x + k * cell, area.y + 20), (area.x + k * cell, area.bottom - 20), 4
        )
        pygame.draw.line(surface, theme.LINE, (area.x + 20, area.y + k * cell), (area.right - 20, area.y + k * cell), 4)
    for index, owner in enumerate(game.cells):
        center = _cell_center(area, index)
        if owner == FIRST:
            _draw_cross(surface, center, cell // 3)
        elif owner == SECOND:
            pygame.draw.circle(surface, theme.BLUE, center, cell // 3, 12)
    line = game.winning_line()
    if line is not None:
        start, end = _cell_center(area, line[0]), _cell_center(area, line[2])
        pygame.draw.line(surface, theme.TEXT, start, end, 8)


def _cell_center(area: pygame.Rect, index: int) -> tuple[int, int]:
    cell = area.width // 3
    return area.x + (index % 3) * cell + cell // 2, area.y + (index // 3) * cell + cell // 2


def _draw_cross(surface: pygame.Surface, center: tuple[int, int], radius: int) -> None:
    x, y = center
    pygame.draw.line(surface, theme.ORANGE, (x - radius, y - radius), (x + radius, y + radius), 14)
    pygame.draw.line(surface, theme.ORANGE, (x - radius, y + radius), (x + radius, y - radius), 14)


def _draw_scoreboard(surface: pygame.Surface, score: Scoreboard) -> None:
    area = pygame.Rect(640, 170, 560, 450)
    drawing.panel(surface, area)
    drawing.text(
        surface, "MINIMAX × ALEATÓRIO", (area.x + 32, area.y + 28), kind="mono_bold", size=18, color=theme.TEXT_SOFT
    )
    columns = (
        ("Vitórias", score.wins, theme.ORANGE),
        ("Empates", score.draws, theme.TEXT),
        ("Derrotas", score.losses, theme.BLUE),
    )
    for i, (label, value, color) in enumerate(columns):
        x = area.x + 32 + i * 175
        drawing.text(surface, str(value), (x, area.y + 62), kind="title", size=72, color=color)
        drawing.text(surface, label, (x, area.y + 150), size=22, color=theme.TEXT_SOFT)
    pygame.draw.line(surface, theme.LINE, (area.x + 32, area.y + 212), (area.right - 32, area.y + 212), 2)
    drawing.text(
        surface, "MINIMAX × MINIMAX", (area.x + 32, area.y + 236), kind="mono_bold", size=18, color=theme.TEXT_SOFT
    )
    drawing.text(
        surface,
        f"{score.self_play_draws} empates em {score.self_play_games} partidas",
        (area.x + 32, area.y + 270),
        kind="title",
        size=34,
    )
    drawing.text(
        surface,
        "O Jogo da Velha é resolvido: com jogo perfeito",
        (area.x + 32, area.y + 340),
        size=22,
        color=theme.TEXT_SOFT,
    )
    drawing.text(
        surface, "dos dois lados, sempre dá empate.", (area.x + 32, area.y + 370), size=22, color=theme.TEXT_SOFT
    )


# ---------------------------------------------------------------- Parte 2: Ligue 4


@dataclass
class DeepMoveStats:
    comparison: PruningComparison | None = None
    total_with_pruning: int = 0
    total_without_pruning: int = 0
    forced_win_in: int | None = None


def connect_four_part(surface: pygame.Surface) -> Scene:
    """Profundidade 2 (azul) começa; profundidade 6 (laranja) responde."""
    game = ConnectFour(current_player=SECOND)
    stats = DeepMoveStats()
    yield from _connect_four_frames(surface, game, stats, 30)

    while not game.is_over():
        if game.current_player == FIRST:
            comparison = compare_pruning(game, DEEP_SEARCH)
            stats.comparison = comparison
            stats.total_with_pruning += comparison.with_pruning.positions_examined
            stats.total_without_pruning += comparison.without_pruning.positions_examined
            stats.forced_win_in = comparison.with_pruning.moves_until_forced_win()
            move = comparison.with_pruning.move
        else:
            move = choose_move(game, SHALLOW_SEARCH).move
        yield from _drop_disc(surface, game, move, stats)
        game = game.play(move)
        pause = 24 if game.current_player == SECOND else 10
        yield from _connect_four_frames(surface, game, stats, pause)

    yield from _connect_four_frames(surface, game, stats, 120)


def _drop_disc(surface: pygame.Surface, game: ConnectFour, column: int, stats: DeepMoveStats) -> Scene:
    target_row = game.landing_row(column)
    for frame in range(1, 9):
        fall = ROWS - 1 + 1.2 - (ROWS - 1 + 1.2 - target_row) * (frame / 8) ** 2
        surface.fill(theme.BACKGROUND)
        _draw_connect_four_screen(surface, game, stats)
        _draw_disc(surface, column, fall, PLAYER_COLORS[game.current_player])
        yield surface


def _connect_four_frames(surface: pygame.Surface, game: ConnectFour, stats: DeepMoveStats, frames: int) -> Scene:
    surface.fill(theme.BACKGROUND)
    _draw_connect_four_screen(surface, game, stats)
    yield from hold(surface, frames / theme.FPS)


BOARD = pygame.Rect(70, 160, 7 * 74 + 20, 6 * 74 + 20)
CELL = 74


def _disc_center(column: int, row: float) -> tuple[float, float]:
    return BOARD.x + 10 + column * CELL + CELL / 2, BOARD.bottom - 10 - row * CELL - CELL / 2


def _draw_disc(surface: pygame.Surface, column: int, row: float, color: theme.Color) -> None:
    pygame.draw.circle(surface, color, _disc_center(column, row), CELL // 2 - 7)


def _draw_connect_four_screen(surface: pygame.Surface, game: ConnectFour, stats: DeepMoveStats) -> None:
    drawing.header(surface, "Parte 2 · Ligue 4", "Quem olha 6 lances à frente × quem olha só 2")
    drawing.panel(surface, BOARD, fill=theme.PANEL, radius=18)
    winning = set(game.winning_window() or ())
    for row in range(ROWS):
        for column in range(COLUMNS):
            owner = game.cells[cell_index(row, column)]
            color = theme.PANEL_DARK if owner == EMPTY else PLAYER_COLORS[owner]
            _draw_disc(surface, column, row, color)
            if cell_index(row, column) in winning:
                pygame.draw.circle(surface, theme.TEXT, _disc_center(column, row), CELL // 2 - 7, 5)
    drawing.legend_item(surface, "laranja: profundidade 6", (BOARD.x, BOARD.bottom + 18), theme.ORANGE)
    drawing.legend_item(surface, "azul: profundidade 2", (BOARD.x + 300, BOARD.bottom + 18), theme.BLUE)
    _draw_pruning_panel(surface, stats)
    _draw_result_banner(surface, game)


def _draw_pruning_panel(surface: pygame.Surface, stats: DeepMoveStats) -> None:
    area = pygame.Rect(680, 160, 560, 484)
    drawing.panel(surface, area)
    x = area.x + 28
    drawing.text(
        surface, "ÚLTIMA JOGADA DA LARANJA", (x, area.y + 24), kind="mono_bold", size=18, color=theme.TEXT_SOFT
    )
    drawing.text(surface, "posições examinadas para decidir", (x, area.y + 52), size=20, color=theme.TEXT_SOFT)
    comparison = stats.comparison
    if comparison is None:
        drawing.text(surface, "aguardando a primeira jogada…", (x, area.y + 110), size=22, color=theme.TEXT_MUTED)
        return

    without = comparison.without_pruning.positions_examined
    with_ = comparison.with_pruning.positions_examined
    drawing.text(surface, "sem poda", (x, area.y + 96), kind="mono", size=18, color=theme.TEXT_SOFT)
    drawing.bar(surface, pygame.Rect(x, area.y + 122, 500, 22), 1.0, theme.TEXT_MUTED)
    drawing.text(surface, drawing.thousands(without), (x, area.y + 150), kind="title", size=30)
    drawing.text(surface, "com poda alfa-beta", (x, area.y + 200), kind="mono", size=18, color=theme.TEXT_SOFT)
    drawing.bar(surface, pygame.Rect(x, area.y + 226, 500, 22), with_ / without, theme.ORANGE)
    drawing.text(surface, drawing.thousands(with_), (x, area.y + 254), kind="title", size=30, color=theme.ORANGE)
    drawing.text(
        surface,
        f"{comparison.savings:.0%} menos · mesma jogada: coluna {comparison.with_pruning.move + 1}",
        (x, area.y + 300),
        size=20,
        color=theme.TEXT,
    )

    pygame.draw.line(surface, theme.LINE, (x, area.y + 344), (area.right - 28, area.y + 344), 2)
    total_savings = 1 - stats.total_with_pruning / stats.total_without_pruning
    drawing.text(
        surface,
        f"Na partida toda: {drawing.thousands(stats.total_without_pruning)} × "
        f"{drawing.thousands(stats.total_with_pruning)}",
        (x, area.y + 360),
        size=20,
        color=theme.TEXT_SOFT,
    )
    drawing.text(
        surface, f"({total_savings:.0%} de cálculo economizado)", (x, area.y + 388), size=20, color=theme.TEXT_SOFT
    )
    if stats.forced_win_in is not None:
        badge = pygame.Rect(x - 8, area.y + 426, 516, 42)
        pygame.draw.rect(surface, theme.ORANGE, badge, border_radius=10)
        plural = "lance" if stats.forced_win_in == 1 else "lances"
        drawing.text(
            surface,
            f"Enxergou vitória garantida em {stats.forced_win_in} {plural}!",
            badge.center,
            size=22,
            color=theme.BACKGROUND,
            anchor="center",
        )


def _draw_result_banner(surface: pygame.Surface, game: ConnectFour) -> None:
    if not game.is_over():
        return
    messages = {FIRST: "Laranja (profundidade 6) venceu", SECOND: "Azul (profundidade 2) venceu"}
    winner = game.winner()
    message = "Empate" if winner is None else messages[winner]
    banner = pygame.Rect(BOARD.x, 668, BOARD.width, 40)
    pygame.draw.rect(surface, theme.TEXT, banner, border_radius=10)
    drawing.text(surface, message, banner.center, kind="title", size=24, color=theme.BACKGROUND, anchor="center")


# ---------------------------------------------------------------- Cena completa


def build_scene() -> Scene:
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    yield from tic_tac_toe_part(surface)
    yield from connect_four_part(surface)
