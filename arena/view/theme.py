"""Paleta e tipografia compartilhadas por todos os vídeos (as mesmas dos slides)."""

from functools import cache
from pathlib import Path

import pygame

Color = tuple[int, int, int]

WIDTH = 1280
HEIGHT = 720
FPS = 30

BACKGROUND: Color = (22, 20, 31)
PANEL: Color = (34, 31, 46)
PANEL_DARK: Color = (14, 13, 20)
LINE: Color = (52, 48, 74)
WALL: Color = (58, 53, 80)
EXPLORED: Color = (36, 86, 128)
TEXT: Color = (243, 238, 228)
TEXT_SOFT: Color = (189, 182, 204)
TEXT_MUTED: Color = (143, 136, 166)
ORANGE: Color = (255, 138, 61)
BLUE: Color = (92, 200, 255)
YELLOW: Color = (255, 201, 60)
RED: Color = (232, 72, 85)
GREEN: Color = (120, 214, 140)

_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_FONT_FILES = {
    "title": "SpaceGrotesk-Bold.ttf",
    "heading": "SpaceGrotesk-Medium.ttf",
    "mono": "JetBrainsMono-Regular.ttf",
    "mono_bold": "JetBrainsMono-Bold.ttf",
}


@cache
def font(kind: str, size: int) -> pygame.font.Font:
    """Fonte do tema. ``kind``: title, heading, mono ou mono_bold."""
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame.font.Font(_FONT_DIR / _FONT_FILES[kind], size)
