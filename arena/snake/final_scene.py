"""Grande final: as seis IAs jogando a MESMA partida de Snake (mesma semente, mesmas maçãs).

É a partida 1 das 100 usadas no placar (``arena.snake.benchmark``).
"""

import pygame

from arena.snake.benchmark import FIRST_SEED
from arena.snake.board_view import draw_board
from arena.snake.contenders import Contender, all_contenders
from arena.snake.game import SnakeGame
from arena.view import drawing, theme
from arena.view.output import Scene, hold

CELL = 22
PANEL_SIZE = (396, 290)


def steps_per_frame(step: int) -> float:
    """Começa devagar para dar para acompanhar e acelera quando sobra pouca gente."""
    if step < 300:
        return 0.5
    if step < 900:
        return 3
    return 12


def build_scene() -> Scene:
    print("Carregando as seis IAs…")
    contenders = all_contenders()
    games = [SnakeGame(seed=FIRST_SEED) for _ in contenders]
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    step, budget = 0, 0.0
    yield from hold(_draw(surface, contenders, games, speed=1), 1.5)
    while not all(game.is_over for game in games):
        speed = steps_per_frame(step)
        budget += speed
        while budget >= 1:
            budget -= 1
            step += 1
            for contender, game in zip(contenders, games, strict=True):
                if not game.is_over:
                    game.step(contender.agent.choose(game))
        yield _draw(surface, contenders, games, speed=max(1, round(speed * 2)))
    yield from hold(_draw(surface, contenders, games, speed=1), 4.0)


def _draw(surface: pygame.Surface, contenders: list[Contender], games: list[SnakeGame], speed: int) -> pygame.Surface:
    surface.fill(theme.BACKGROUND)
    drawing.header(surface, "Grande final · Snake 10 × 10", "Seis IAs, a mesma partida")
    if speed > 1:
        badge = pygame.Rect(theme.WIDTH - 230, 40, 190, 36)
        pygame.draw.rect(surface, theme.BLUE, badge, border_radius=8)
        drawing.text(
            surface,
            f"{speed}× mais rápido",
            badge.center,
            kind="mono_bold",
            size=18,
            color=theme.BACKGROUND,
            anchor="center",
        )
    for index, (contender, game) in enumerate(zip(contenders, games, strict=True)):
        column, row = index % 3, index // 3
        _draw_panel(surface, contender, game, index + 1, (32 + column * 412, 104 + row * 304))
    return surface


def _draw_panel(
    surface: pygame.Surface, contender: Contender, game: SnakeGame, number: int, origin: tuple[int, int]
) -> None:
    x, y = origin
    drawing.panel(surface, pygame.Rect(x, y, *PANEL_SIZE))
    drawing.text(surface, f"{number} · {contender.agent.name}", (x + 16, y + 12), kind="title", size=20)
    draw_board(surface, game, (x + 21, y + 54), CELL)
    right = x + 268
    drawing.text(surface, "MAÇÃS", (right, y + 60), kind="mono_bold", size=16, color=theme.TEXT_SOFT)
    drawing.text(surface, str(game.score), (right, y + 80), kind="title", size=56, color=theme.ORANGE)
    drawing.text(surface, "PASSOS", (right, y + 160), kind="mono_bold", size=16, color=theme.TEXT_SOFT)
    drawing.text(surface, drawing.thousands(game.steps), (right, y + 180), kind="title", size=26)
    status, color = _status(game)
    drawing.text(surface, status, (right, y + 236), kind="mono_bold", size=16, color=color)


def _status(game: SnakeGame) -> tuple[str, theme.Color]:
    if game.won:
        return "zerou!", theme.GREEN
    if not game.alive:
        return "fora do jogo", theme.RED
    return "jogando", theme.TEXT_SOFT
