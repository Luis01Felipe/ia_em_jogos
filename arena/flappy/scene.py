"""Vídeo 3: a população de pássaros evoluindo, da geração 1 até dominar o jogo.

Todas as gerações mostradas são reprises exatas do treino (mesma população, mesmos canos).
"""

import numpy as np
import pygame

from arena.flappy.evolution import LAYERS, GenerationRecord, evolve, run_generation
from arena.flappy.world import FlappyConfig, FlappyWorld
from arena.shared.neural_net import layer_activations
from arena.view import drawing, theme
from arena.view.output import Scene, hold

SEED = 1
GENERATIONS = 25
TIME_LAPSE_GENERATIONS = range(2, 9)
CONFIG = FlappyConfig()
GAME_AREA = pygame.Rect(40, 112, 760, 570)
SCALE = GAME_AREA.width / CONFIG.width
SIDE = pygame.Rect(830, 112, 410, 570)
ROUND_SECONDS = CONFIG.max_frames // theme.FPS  # sobreviver tudo isso = zerar a rodada


def build_scene() -> Scene:
    print("Treinando a população (leva poucos segundos)…")
    history = evolve(GENERATIONS, SEED, CONFIG)
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))

    first, last = history[0], history[-1]
    yield from _replay(surface, history, first, speed=1, max_video_frames=None)
    for record in history[1:]:
        if record.number in TIME_LAPSE_GENERATIONS:
            yield from _replay(surface, history, record, speed=4, max_video_frames=45)
    yield from _skip_ahead(surface, history, last)
    yield from _replay_champions(surface, history, last)


def _replay(
    surface: pygame.Surface,
    history: list[GenerationRecord],
    record: GenerationRecord,
    *,
    speed: int,
    max_video_frames: int | None,
) -> Scene:
    frames = run_generation(record.population, CONFIG, record.pipe_seed)
    shown = 0
    world = next(frames)
    for index, world in enumerate(frames):
        if index % speed:
            continue
        _draw(surface, history, record, world, speed)
        yield surface
        shown += 1
        if max_video_frames is not None and shown >= max_video_frames:
            return
    yield from hold(_draw(surface, history, record, world, speed), 0.7)


def _skip_ahead(surface: pygame.Surface, history: list[GenerationRecord], target: GenerationRecord) -> Scene:
    """Avança o gráfico geração por geração até a última, sem mostrar o jogo."""
    for record in history[TIME_LAPSE_GENERATIONS.stop - 1 : target.number - 1]:
        surface.fill(theme.BACKGROUND)
        drawing.header(surface, "Rodada 3 · Neuroevolução", f"Pulando para a geração {target.number}…")
        drawing.panel(surface, GAME_AREA, fill=theme.PANEL_DARK)
        drawing.text(
            surface,
            f"geração {record.number}",
            GAME_AREA.center,
            kind="title",
            size=64,
            color=theme.TEXT_MUTED,
            anchor="center",
        )
        _draw_side_panel(surface, history, record)
        yield from hold(surface, 0.12)


def _replay_champions(surface: pygame.Surface, history: list[GenerationRecord], record: GenerationRecord) -> Scene:
    """Última geração: 8 s em tempo real e depois acelerado até o fim da rodada."""
    frames = run_generation(record.population, CONFIG, record.pipe_seed)
    world = next(frames)
    for index, world in enumerate(frames):
        realtime = index < 8 * theme.FPS
        if not realtime and index % 8:
            continue
        _draw(surface, history, record, world, speed=1 if realtime else 8, show_brain=True)
        yield surface
    yield from hold(_draw(surface, history, record, world, speed=1, show_brain=True), 3.0)


def _draw(
    surface: pygame.Surface,
    history: list[GenerationRecord],
    record: GenerationRecord,
    world: FlappyWorld,
    speed: int,
    show_brain: bool = False,
) -> pygame.Surface:
    surface.fill(theme.BACKGROUND)
    title = "Geração 1: cérebros aleatórios" if record.number == 1 else f"Geração {record.number}"
    drawing.header(surface, "Rodada 3 · Neuroevolução no Flappy Bird", title)
    _draw_world(surface, world)
    _draw_hud(surface, world, speed)
    if show_brain:
        _draw_brain(surface, record, world)
    else:
        _draw_side_panel(surface, history, record)
    return surface


def _draw_world(surface: pygame.Surface, world: FlappyWorld) -> None:
    drawing.panel(surface, GAME_AREA, fill=theme.PANEL_DARK, border=None)
    clip = surface.get_clip()
    surface.set_clip(GAME_AREA)
    half_gap = CONFIG.pipe_gap / 2
    for pipe in world.pipes:
        x = GAME_AREA.x + pipe.x * SCALE
        width = CONFIG.pipe_width * SCALE
        top_height = (pipe.gap_center - half_gap) * SCALE
        bottom_y = GAME_AREA.y + (pipe.gap_center + half_gap) * SCALE
        pygame.draw.rect(surface, theme.WALL, (x, GAME_AREA.y, width, top_height))
        pygame.draw.rect(surface, theme.WALL, (x, bottom_y, width, GAME_AREA.bottom - bottom_y))
    bird_x = GAME_AREA.x + CONFIG.bird_x * SCALE
    radius = CONFIG.bird_radius * SCALE
    birds = pygame.Surface(GAME_AREA.size, pygame.SRCALPHA)
    for y in world.flock.y[world.flock.alive]:
        pygame.draw.circle(birds, (*theme.ORANGE, 150), (bird_x - GAME_AREA.x, y * SCALE), radius)
    surface.blit(birds, GAME_AREA.topleft)
    surface.set_clip(clip)


def _draw_hud(surface: pygame.Surface, world: FlappyWorld, speed: int) -> None:
    alive = int(world.flock.alive.sum())
    right = GAME_AREA.right - 20
    drawing.text(
        surface,
        f"vivos: {alive}/{len(world.flock.alive)}",
        (right, GAME_AREA.y + 16),
        kind="mono_bold",
        size=22,
        color=theme.TEXT,
        anchor="topright",
    )
    drawing.text(
        surface,
        f"canos: {world.pipes_passed}",
        (right, GAME_AREA.y + 46),
        kind="mono_bold",
        size=22,
        color=theme.ORANGE,
        anchor="topright",
    )
    if speed > 1:
        badge = pygame.Rect(right - 170, GAME_AREA.y + 84, 170, 34)
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


def _draw_side_panel(surface: pygame.Surface, history: list[GenerationRecord], record: GenerationRecord) -> None:
    drawing.panel(surface, SIDE)
    seen = history[: record.number]
    chart = pygame.Rect(SIDE.x + 16, SIDE.y + 16, SIDE.width - 32, 300)
    y_max = max(r.best_pipes for r in history)
    drawing.line_chart(
        surface,
        chart,
        [r.best_pipes for r in seen],
        y_max=y_max,
        title="melhor pássaro (canos) por geração",
        x_label=f"geração {record.number}",
    )
    x = SIDE.x + 28
    drawing.text(surface, "MELHOR DESTA GERAÇÃO", (x, SIDE.y + 340), kind="mono_bold", size=18, color=theme.TEXT_SOFT)
    drawing.text(surface, _pipes(record.best_pipes), (x, SIDE.y + 366), kind="title", size=44, color=theme.ORANGE)
    drawing.text(
        surface,
        f"ZERARAM ({ROUND_SECONDS} S SEM MORRER)",
        (x, SIDE.y + 440),
        kind="mono_bold",
        size=18,
        color=theme.TEXT_SOFT,
    )
    drawing.text(
        surface,
        f"{record.birds_finished} de {len(record.population)} pássaros",
        (x, SIDE.y + 466),
        kind="title",
        size=34,
    )
    drawing.text(surface, "nota de cada pássaro = tempo vivo", (x, SIDE.bottom - 40), size=18, color=theme.TEXT_MUTED)


def _draw_brain(surface: pygame.Surface, record: GenerationRecord, world: FlappyWorld) -> None:
    """Desenha a rede de um pássaro vivo: entradas, neurônios ocultos e a decisão."""
    drawing.panel(surface, SIDE)
    x = SIDE.x + 28
    drawing.text(surface, "O CÉREBRO DE UM CAMPEÃO", (x, SIDE.y + 24), kind="mono_bold", size=18, color=theme.TEXT_SOFT)
    drawing.text(surface, "21 números, nenhum escrito à mão", (x, SIDE.y + 50), size=20, color=theme.TEXT_SOFT)
    bird = _first_alive(world)
    if bird is None:
        return
    layers = layer_activations(record.population[bird], world.sensors()[bird], LAYERS)
    positions = _neuron_positions()
    for left, right in zip(positions, positions[1:], strict=False):
        for start in left:
            for end in right:
                pygame.draw.line(surface, theme.LINE, start, end, 2)
    for values, column in zip(layers, positions, strict=True):
        for value, center in zip(values, column, strict=True):
            _draw_neuron(surface, center, float(value))
    labels = ("distância", "altura do buraco", "velocidade")
    for label, (cx, cy) in zip(labels, positions[0], strict=True):
        drawing.text(surface, label, (cx - 26, cy), size=18, color=theme.TEXT_SOFT, anchor="midright")
    flap = layers[-1][0] > 0
    out_x, out_y = positions[-1][0]
    drawing.text(
        surface,
        "PULA!" if flap else "espera",
        (out_x, out_y + 40),
        kind="title",
        size=26,
        color=theme.ORANGE if flap else theme.TEXT_MUTED,
        anchor="midtop",
    )
    drawing.text(
        surface,
        f"{record.birds_finished} de {len(record.population)} pássaros desta geração",
        (x, SIDE.bottom - 76),
        size=20,
        color=theme.TEXT,
    )
    drawing.text(surface, f"zeraram: {ROUND_SECONDS} s sem morrer", (x, SIDE.bottom - 48), size=20, color=theme.TEXT)


def _pipes(count: int) -> str:
    return "1 cano" if count == 1 else f"{count} canos"


def _neuron_positions() -> list[list[tuple[int, int]]]:
    columns_x = (SIDE.x + 180, SIDE.x + 270, SIDE.x + 350)
    center_y = SIDE.y + 270
    positions = []
    for size, cx in zip(LAYERS, columns_x, strict=True):
        spacing = 70
        top = center_y - spacing * (size - 1) / 2
        positions.append([(cx, round(top + i * spacing)) for i in range(size)])
    return positions


def _draw_neuron(surface: pygame.Surface, center: tuple[int, int], value: float) -> None:
    intensity = min(abs(value), 1.0)
    base = theme.ORANGE if value > 0 else theme.BLUE
    color = tuple(round(theme.PANEL_DARK[i] + (base[i] - theme.PANEL_DARK[i]) * intensity) for i in range(3))
    pygame.draw.circle(surface, color, center, 20)
    pygame.draw.circle(surface, theme.TEXT_MUTED, center, 20, 2)


def _first_alive(world: FlappyWorld) -> int | None:
    alive = np.flatnonzero(world.flock.alive)
    return int(alive[0]) if alive.size else None
