"""Primitivas de desenho reaproveitadas pelas cenas (texto, painéis e gráficos)."""

from collections.abc import Sequence

import pygame

from arena.shared.formatting import thousands  # noqa: F401  (reexportado para as cenas)
from arena.view import theme
from arena.view.theme import Color


def text(
    surface: pygame.Surface,
    content: str,
    pos: tuple[float, float],
    *,
    kind: str = "heading",
    size: int = 24,
    color: Color = theme.TEXT,
    anchor: str = "topleft",
) -> pygame.Rect:
    """Escreve ``content`` ancorado em ``pos`` e devolve o retângulo ocupado."""
    rendered = theme.font(kind, size).render(content, True, color)
    rect = rendered.get_rect(**{anchor: (round(pos[0]), round(pos[1]))})
    surface.blit(rendered, rect)
    return rect


def panel(
    surface: pygame.Surface,
    rect: pygame.Rect,
    *,
    fill: Color = theme.PANEL,
    border: Color | None = theme.LINE,
    radius: int = 14,
    width: int = 2,
) -> None:
    pygame.draw.rect(surface, fill, rect, border_radius=radius)
    if border is not None:
        pygame.draw.rect(surface, border, rect, width=width, border_radius=radius)


def header(surface: pygame.Surface, eyebrow: str, title: str) -> None:
    """Cabeçalho padrão de todos os vídeos: rótulo pequeno + título."""
    text(surface, eyebrow.upper(), (40, 28), kind="mono_bold", size=18, color=theme.ORANGE)
    text(surface, title, (40, 52), kind="title", size=34)


def line_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    values: Sequence[float],
    *,
    y_max: float,
    color: Color = theme.ORANGE,
    title: str = "",
    x_label: str = "",
) -> None:
    """Gráfico de linha simples, com o último valor destacado."""
    panel(surface, rect, fill=theme.PANEL_DARK)
    if title:
        text(surface, title, (rect.x + 16, rect.y + 10), kind="mono", size=16, color=theme.TEXT_SOFT)
    plot = pygame.Rect(rect.x + 16, rect.y + 40, rect.width - 32, rect.height - 72)
    pygame.draw.line(surface, theme.LINE, plot.bottomleft, plot.bottomright, 2)
    if x_label:
        text(
            surface,
            x_label,
            (plot.right, rect.bottom - 10),
            kind="mono",
            size=14,
            color=theme.TEXT_MUTED,
            anchor="bottomright",
        )
    if len(values) < 2 or y_max <= 0:
        return
    step = plot.width / (len(values) - 1)
    points = [
        (plot.x + i * step, plot.bottom - min(value / y_max, 1.0) * plot.height) for i, value in enumerate(values)
    ]
    pygame.draw.lines(surface, color, False, points, 3)
    pygame.draw.circle(surface, color, points[-1], 6)


def bar(
    surface: pygame.Surface,
    rect: pygame.Rect,
    fraction: float,
    color: Color,
) -> None:
    """Barra horizontal preenchida até ``fraction`` (0 a 1)."""
    pygame.draw.rect(surface, theme.PANEL_DARK, rect, border_radius=6)
    filled = rect.copy()
    filled.width = max(4, round(rect.width * max(0.0, min(fraction, 1.0))))
    pygame.draw.rect(surface, color, filled, border_radius=6)


def legend_item(surface: pygame.Surface, label: str, pos: tuple[float, float], color: Color) -> None:
    """Bolinha colorida seguida de um rótulo (legenda de gráfico ou de jogador)."""
    x, y = pos
    pygame.draw.circle(surface, color, (round(x + 8), round(y + 13)), 8)
    text(surface, label, (x + 24, y), size=20, color=color)
