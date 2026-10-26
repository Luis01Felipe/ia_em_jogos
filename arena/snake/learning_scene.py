"""Vídeo 5: a cobra aprendendo por reforço (Q-Learning) e o que acontece quando a recompensa muda.

Parte 1 — a mesma IA em três momentos do treino, jogando lado a lado.
Parte 2 — duas IAs treinadas com recompensas diferentes.
"""

from dataclasses import dataclass, field

import numpy as np
import pygame

from arena.snake import qlearning
from arena.snake.board_view import draw_board
from arena.snake.game import SnakeGame
from arena.snake.qlearning import EAT_APPLES, STAY_ALIVE, QLearningAgent
from arena.view import drawing, theme
from arena.view.output import Scene, hold

TRAINING_EPISODES = 3_000
CHECKPOINTS = (0, 100, TRAINING_EPISODES)
HACK_EPISODES = 300
FRAMES_PER_STEP = 2
CELL = 30
FIRST_SHOW_SEED = 20_000


@dataclass
class Showcase:
    """Uma IA jogando partida após partida num quadro do vídeo."""

    title: str
    agent: QLearningAgent
    starvation_limit: int | None = 200
    game_number: int = 0
    finished_scores: list[int] = field(default_factory=list)
    game: SnakeGame = field(init=False)
    frames_dead: int = 0

    def __post_init__(self) -> None:
        self._new_game()

    def advance(self) -> None:
        if self.game.is_over:
            self.frames_dead += 1
            if self.frames_dead > 30:
                self.finished_scores.append(self.game.score)
                self._new_game()
            return
        self.game.step(self.agent.choose(self.game))

    def _new_game(self) -> None:
        self.game_number += 1
        self.frames_dead = 0
        self.game = SnakeGame(seed=FIRST_SHOW_SEED + self.game_number, starvation_limit=self.starvation_limit)


def build_scene() -> Scene:
    print("Treinando a tabela Q (cerca de 20 segundos)…")
    training = qlearning.train(TRAINING_EPISODES, snapshot_at=CHECKPOINTS)
    hacked = qlearning.train(HACK_EPISODES, rewards=STAY_ALIVE, starvation_limit=None)
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    yield from _three_moments(surface, training, seconds=26)
    yield from _reward_comparison(surface, training.table, hacked.table, seconds=16)


def _three_moments(surface: pygame.Surface, training: qlearning.TrainingResult, seconds: int) -> Scene:
    labels = {
        0: "Antes do treino (tabela vazia)",
        100: "Depois de 100 partidas",
        TRAINING_EPISODES: "Depois de 3.000 partidas",
    }
    showcases = [Showcase(labels[ep], QLearningAgent(training.snapshots[ep], seed=ep)) for ep in CHECKPOINTS]
    curve = _moving_average(training.scores, window=100)
    for frame in range(seconds * theme.FPS):
        if frame % FRAMES_PER_STEP == 0:
            for showcase in showcases:
                showcase.advance()
        surface.fill(theme.BACKGROUND)
        drawing.header(surface, "Rodada 5 · Q-Learning no Snake", "A mesma IA em três momentos do treino")
        for index, showcase in enumerate(showcases):
            _draw_showcase(surface, showcase, (90 + index * 405, 150))
        _draw_curve(surface, curve)
        yield surface


def _reward_comparison(
    surface: pygame.Surface, apples_table: np.ndarray, survival_table: np.ndarray, seconds: int
) -> Scene:
    showcases = [
        Showcase(f"Recompensa: {EAT_APPLES.name}", QLearningAgent(apples_table, seed=1)),
        Showcase(f"Recompensa: {STAY_ALIVE.name}", QLearningAgent(survival_table, seed=1), starvation_limit=None),
    ]
    for frame in range(seconds * theme.FPS):
        if frame % FRAMES_PER_STEP == 0:
            for showcase in showcases:
                showcase.advance()
        surface.fill(theme.BACKGROUND)
        drawing.header(
            surface, "Rodada 5 · O que você recompensa é o que você recebe", "Mesmo algoritmo, recompensas diferentes"
        )
        for index, showcase in enumerate(showcases):
            _draw_showcase(surface, showcase, (230 + index * 520, 150), show_steps=True)
        message = "Pedimos para ficar viva, não para comer: ela dá voltas para sempre e ignora a maçã."
        drawing.text(surface, message, (theme.WIDTH / 2, 600), size=24, color=theme.TEXT, anchor="midtop")
        yield surface
    yield from hold(surface, 1.5)


def _draw_showcase(
    surface: pygame.Surface, showcase: Showcase, origin: tuple[int, int], show_steps: bool = False
) -> None:
    x, y = origin
    drawing.text(surface, showcase.title, (x - 5, y - 44), kind="title", size=22)
    board = draw_board(surface, showcase.game, (x, y), CELL)
    status_y = board.bottom + 16
    drawing.text(
        surface,
        f"partida {showcase.game_number} · maçãs: {showcase.game.score}",
        (x - 5, status_y),
        kind="mono_bold",
        size=18,
        color=theme.ORANGE,
    )
    if show_steps:
        drawing.text(
            surface,
            f"passos vivo: {showcase.game.steps}",
            (x - 5, status_y + 28),
            kind="mono",
            size=18,
            color=theme.TEXT_SOFT,
        )
        return
    if showcase.finished_scores:
        mean = sum(showcase.finished_scores) / len(showcase.finished_scores)
        drawing.text(
            surface,
            f"média das partidas: {mean:.1f}".replace(".", ","),
            (x - 5, status_y + 28),
            kind="mono",
            size=18,
            color=theme.TEXT_SOFT,
        )


def _draw_curve(surface: pygame.Surface, curve: list[float]) -> None:
    area = pygame.Rect(90, 540, 1110, 160)
    drawing.line_chart(
        surface,
        area,
        curve,
        y_max=max(curve) * 1.1,
        title="maçãs por partida durante o treino (média móvel de 100 partidas)",
        x_label=f"partida {drawing.thousands(len(curve))}",
    )


def _moving_average(values: list[int], window: int) -> list[float]:
    averages, total = [], 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        averages.append(total / min(index + 1, window))
    return averages
