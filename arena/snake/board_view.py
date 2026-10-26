"""Desenho de um tabuleiro de Snake (compartilhado pelos vídeos da Rodada 5 e do grande final)."""

import pygame

from arena.snake.game import SnakeGame
from arena.view import drawing, theme

DEATH_LABELS = {"bateu": "bateu", "fome": "morreu de fome"}


def draw_board(surface: pygame.Surface, game: SnakeGame, origin: tuple[int, int], cell: int) -> pygame.Rect:
    x0, y0 = origin
    board = pygame.Rect(x0, y0, game.size * cell, game.size * cell)
    pygame.draw.rect(surface, theme.PANEL_DARK, board.inflate(10, 10), border_radius=8)
    pygame.draw.rect(surface, theme.LINE, board.inflate(10, 10), width=2, border_radius=8)
    for gx in range(game.size):
        for gy in range(game.size):
            center = (x0 + gx * cell + cell // 2, y0 + gy * cell + cell // 2)
            pygame.draw.circle(surface, theme.LINE, center, 2)

    if game.apple is not None:
        ax, ay = game.apple
        pygame.draw.circle(surface, theme.BLUE, (x0 + ax * cell + cell // 2, y0 + ay * cell + cell // 2), cell * 0.38)

    length = len(game.body)
    for index, (bx, by) in enumerate(game.body):
        fade = 1.0 if index == 0 else 0.55 + 0.4 * (1 - index / length)
        color = tuple(round(c * fade + theme.PANEL_DARK[i] * (1 - fade)) for i, c in enumerate(theme.ORANGE))
        rect = pygame.Rect(x0 + bx * cell + 1, y0 + by * cell + 1, cell - 2, cell - 2)
        pygame.draw.rect(surface, color, rect, border_radius=max(2, cell // 5))

    if not game.alive:
        veil = pygame.Surface(board.size, pygame.SRCALPHA)
        veil.fill((*theme.BACKGROUND, 170))
        surface.blit(veil, board.topleft)
        label = DEATH_LABELS.get(game.cause_of_death, "fim")
        drawing.text(surface, label, board.center, kind="title", size=max(20, cell), color=theme.RED, anchor="center")
    elif game.won:
        pill = pygame.Rect(0, 0, board.width * 0.7, cell * 2)
        pill.center = board.center
        pygame.draw.rect(surface, theme.BACKGROUND, pill, border_radius=10)
        drawing.text(
            surface, "ZEROU!", board.center, kind="title", size=max(24, cell + 6), color=theme.GREEN, anchor="center"
        )
    return board
