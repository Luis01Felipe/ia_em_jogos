"""Vídeo 4: a IA aprende Pong só olhando exemplos de um professor (e copia até os vícios dele).

Também tem um modo para você mesmo ser o professor:
``python -m arena.pong --gravar minhas_partidas.npz`` e depois ``--imitar minhas_partidas.npz``.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pygame

from arena.pong.game import Action, PongConfig, PongGame, PongState
from arena.pong.imitation import Dataset, MatchReport, TrainedStudent, evaluate, record, train
from arena.pong.players import GoodTeacher, HabitTeacher, Paddle, observation
from arena.view import drawing, theme
from arena.view.output import Scene, hold

CONFIG = PongConfig()
RECORDING_SEED = 1
EVALUATION_SEED = 99
TRAINING_FRAMES = 20_000
EVALUATION_FRAMES = 9_000
TEACHER_COLOR = theme.BLUE
STUDENT_COLOR = theme.ORANGE
ACTION_NAMES = {Action.STAY: "parado", Action.UP: "sobe", Action.DOWN: "desce"}


@dataclass(frozen=True)
class Lesson:
    """Um professor, o aluno treinado com os exemplos dele e as medições dos dois."""

    teacher: Paddle
    trained: TrainedStudent
    teacher_report: MatchReport
    student_report: MatchReport


def prepare_lesson(teacher: Paddle) -> Lesson:
    dataset = record(teacher, TRAINING_FRAMES, CONFIG, RECORDING_SEED)
    trained = train(dataset, CONFIG)
    return Lesson(
        teacher,
        trained,
        evaluate(teacher, EVALUATION_FRAMES, CONFIG, EVALUATION_SEED),
        evaluate(trained.student, EVALUATION_FRAMES, CONFIG, EVALUATION_SEED),
    )


# ---------------------------------------------------------------- vídeo da palestra


def build_scene() -> Scene:
    print("Gravando os professores e treinando os alunos…")
    good = prepare_lesson(GoodTeacher(CONFIG))
    habit = prepare_lesson(HabitTeacher(CONFIG))
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    yield from _teacher_recording(surface, good, seconds=12)
    yield from _training_card(surface, good)
    yield from _student_playing(surface, good, seconds=14)
    yield from _habit_side_by_side(surface, habit, seconds=16)


def _teacher_recording(surface: pygame.Surface, lesson: Lesson, seconds: int) -> Scene:
    """Os primeiros quadros da gravação de verdade (mesma semente usada no treino)."""
    game = PongGame(CONFIG, RECORDING_SEED)
    recent: list[tuple[np.ndarray, Action]] = []
    area = pygame.Rect(40, 130, 720, 450)
    for frame in range(seconds * theme.FPS):
        action = lesson.teacher.act(game.state)
        recent = (recent + [(observation(game.state, CONFIG), action)])[-8:]
        game.step(action)
        surface.fill(theme.BACKGROUND)
        drawing.header(surface, "Rodada 4 · Imitação no Pong", "1. O professor joga — cada quadro vira um exemplo")
        _draw_game(surface, game.state, area, TEACHER_COLOR, "professor")
        _draw_examples(surface, recent, frame + 1)
        yield surface


def _training_card(surface: pygame.Surface, lesson: Lesson) -> Scene:
    trained = lesson.trained
    minutes = trained.examples / theme.FPS / 60
    surface.fill(theme.BACKGROUND)
    drawing.header(surface, "Rodada 4 · Imitação no Pong", "2. Treino")
    lines = (
        (f"{drawing.thousands(trained.examples)} exemplos gravados", theme.TEXT, 44),
        (f"(cerca de {minutes:.0f} minutos de jogo do professor)", theme.TEXT_SOFT, 26),
        ("Uma rede neural aprende a prever a tecla do professor", theme.TEXT, 30),
        (f"Acerto em exemplos que ela nunca viu: {trained.test_accuracy:.1%}".replace(".", ","), theme.ORANGE, 30),
    )
    y = 220
    for content, color, size in lines:
        drawing.text(surface, content, (theme.WIDTH / 2, y), kind="title", size=size, color=color, anchor="midtop")
        y += size + 40
    yield from hold(surface, 4.0)


def _student_playing(surface: pygame.Surface, lesson: Lesson, seconds: int) -> Scene:
    game = PongGame(CONFIG, EVALUATION_SEED)
    student = lesson.trained.student
    area = pygame.Rect(40, 130, 720, 450)
    for _ in range(seconds * theme.FPS):
        game.step(student.act(game.state))
        surface.fill(theme.BACKGROUND)
        drawing.header(
            surface, "Rodada 4 · Imitação no Pong", "3. A IA assume a raquete (nunca viu as regras do professor)"
        )
        _draw_game(surface, game.state, area, STUDENT_COLOR, "IA (aluno)")
        _draw_student_panel(surface, game.state, lesson)
        yield surface


def _habit_side_by_side(surface: pygame.Surface, lesson: Lesson, seconds: int) -> Scene:
    teacher_game = PongGame(CONFIG, EVALUATION_SEED)
    student_game = PongGame(CONFIG, EVALUATION_SEED)
    student = lesson.trained.student
    left_area, right_area = pygame.Rect(40, 170, 580, 362), pygame.Rect(660, 170, 580, 362)
    for _ in range(seconds * theme.FPS):
        teacher_game.step(lesson.teacher.act(teacher_game.state))
        student_game.step(student.act(student_game.state))
        surface.fill(theme.BACKGROUND)
        drawing.header(
            surface, "Rodada 4 · Imitação no Pong", "4. Um professor com vício: depois de rebater, sobe até o topo"
        )
        _draw_game(surface, teacher_game.state, left_area, TEACHER_COLOR, "professor com vício", show_top_zone=True)
        _draw_game(surface, student_game.state, right_area, STUDENT_COLOR, "IA que imitou ele", show_top_zone=True)
        _draw_habit_summary(surface, lesson, left_area, right_area)
        yield surface


# ---------------------------------------------------------------- modo "você é o professor"


def build_recording_scene(output: Path) -> Scene:
    """Você joga com W/S ou com as setas. Ao fechar a janela (ou ESC), os exemplos são salvos."""
    from arena.pong.keyboard import KeyboardPaddle  # só o modo interativo precisa do teclado

    game = PongGame(CONFIG, RECORDING_SEED)
    you = KeyboardPaddle()
    observations: list[np.ndarray] = []
    actions: list[int] = []
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    area = pygame.Rect(40, 130, 720, 450)
    try:
        while True:
            action = you.act(game.state)
            observations.append(observation(game.state, CONFIG))
            actions.append(int(action))
            game.step(action)
            surface.fill(theme.BACKGROUND)
            drawing.header(surface, "Você é o professor", "W/S ou setas para mexer · ESC para terminar e salvar")
            _draw_game(surface, game.state, area, TEACHER_COLOR, "você")
            drawing.text(
                surface,
                f"exemplos gravados: {drawing.thousands(len(actions))}",
                (800, 140),
                kind="mono_bold",
                size=22,
                color=theme.BLUE,
            )
            yield surface
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, observations=np.array(observations), actions=np.array(actions))
        print(f"{len(actions)} exemplos salvos em {output}")


def build_imitation_scene(dataset_path: Path) -> Scene:
    """Treina um aluno com os seus exemplos e mostra ele jogando."""
    data = np.load(dataset_path)
    trained = train(Dataset(data["observations"], data["actions"]), CONFIG)
    print(f"Aluno treinado com {trained.examples} exemplos; acerto em exemplos novos: {trained.test_accuracy:.1%}")
    game = PongGame(CONFIG, EVALUATION_SEED)
    surface = pygame.Surface((theme.WIDTH, theme.HEIGHT))
    area = pygame.Rect(40, 130, 720, 450)
    while True:
        game.step(trained.student.act(game.state))
        surface.fill(theme.BACKGROUND)
        drawing.header(
            surface, "A IA que imitou você", f"treinada com {drawing.thousands(trained.examples)} exemplos seus"
        )
        _draw_game(surface, game.state, area, STUDENT_COLOR, "IA (aluno)")
        drawing.text(surface, _hits_label(game.state), (800, 140), kind="mono_bold", size=22, color=theme.ORANGE)
        yield surface


# ---------------------------------------------------------------- desenho


def _draw_game(
    surface: pygame.Surface,
    state: PongState,
    area: pygame.Rect,
    paddle_color: theme.Color,
    label: str,
    show_top_zone: bool = False,
) -> None:
    scale = area.width / CONFIG.width
    drawing.panel(surface, area, fill=theme.PANEL_DARK, border=theme.LINE, radius=10)
    if show_top_zone:
        zone = pygame.Surface((area.width, round((CONFIG.paddle_height + 15) * scale)), pygame.SRCALPHA)
        zone.fill((*theme.ORANGE, 40))
        surface.blit(zone, area.topleft)
    for y in range(area.y + 8, area.bottom - 8, 24):
        pygame.draw.line(surface, theme.LINE, (area.centerx, y), (area.centerx, y + 12), 2)

    def paddle_rect(x: float, center: float) -> pygame.Rect:
        top = center - CONFIG.paddle_height / 2
        return pygame.Rect(
            area.x + x * scale, area.y + top * scale, CONFIG.paddle_width * scale, CONFIG.paddle_height * scale
        )

    pygame.draw.rect(surface, paddle_color, paddle_rect(CONFIG.paddle_margin, state.left_y), border_radius=4)
    right_x = CONFIG.width - CONFIG.paddle_margin - CONFIG.paddle_width
    pygame.draw.rect(surface, theme.TEXT_MUTED, paddle_rect(right_x, state.right_y), border_radius=4)
    ball = (area.x + state.ball_x * scale, area.y + state.ball_y * scale)
    pygame.draw.circle(surface, theme.TEXT, ball, CONFIG.ball_radius * scale)
    drawing.text(surface, label, (area.x, area.bottom + 12), size=22, color=paddle_color)
    drawing.text(
        surface,
        _hits_label(state),
        (area.right, area.bottom + 12),
        kind="mono",
        size=20,
        color=theme.TEXT_SOFT,
        anchor="topright",
    )


def _hits_label(state: PongState) -> str:
    return f"rebatidas: {state.left_hits} · perdidas: {state.left_misses}"


def _draw_examples(surface: pygame.Surface, recent: list[tuple[np.ndarray, Action]], total: int) -> None:
    area = pygame.Rect(790, 130, 450, 450)
    drawing.panel(surface, area)
    x = area.x + 24
    drawing.text(surface, "EXEMPLOS GRAVADOS", (x, area.y + 22), kind="mono_bold", size=18, color=theme.TEXT_SOFT)
    drawing.text(surface, drawing.thousands(total), (x, area.y + 46), kind="title", size=48, color=theme.BLUE)
    drawing.text(
        surface, "bola x  bola y  raquete  =>  tecla", (x, area.y + 124), kind="mono", size=16, color=theme.TEXT_MUTED
    )
    for row, (features, action) in enumerate(reversed(recent)):
        ball_x, ball_y, *_, paddle_y, _gap = features
        line = f"{ball_x:5.2f}   {ball_y:5.2f}   {paddle_y:5.2f}   =>  {ACTION_NAMES[action]}"
        color = theme.TEXT if row == 0 else theme.TEXT_SOFT
        drawing.text(surface, line, (x, area.y + 154 + row * 32), kind="mono", size=18, color=color)
    drawing.text(
        surface, "(+ velocidade da bola, em cada exemplo)", (x, area.bottom - 34), size=16, color=theme.TEXT_MUTED
    )


def _draw_student_panel(surface: pygame.Surface, state: PongState, lesson: Lesson) -> None:
    area = pygame.Rect(790, 130, 450, 450)
    drawing.panel(surface, area)
    x = area.x + 24
    drawing.text(
        surface, "MEDIÇÃO EM 5 MINUTOS DE JOGO", (x, area.y + 22), kind="mono_bold", size=18, color=theme.TEXT_SOFT
    )
    rows = (
        ("professor", lesson.teacher_report.hit_rate, TEACHER_COLOR),
        ("IA (aluno)", lesson.student_report.hit_rate, STUDENT_COLOR),
    )
    for index, (name, rate, color) in enumerate(rows):
        y = area.y + 64 + index * 100
        drawing.text(surface, f"{name}: {rate:.0%} das bolas rebatidas", (x, y), size=22, color=color)
        drawing.bar(surface, pygame.Rect(x, y + 36, 400, 20), rate, color)
    drawing.text(surface, "O aluno só viu exemplos de", (x, area.y + 290), size=22, color=theme.TEXT)
    drawing.text(surface, "(situação => tecla). Joga como", (x, area.y + 320), size=22, color=theme.TEXT)
    drawing.text(surface, "o professor, nem melhor nem pior.", (x, area.y + 350), size=22, color=theme.TEXT)


def _draw_habit_summary(surface: pygame.Surface, lesson: Lesson, left: pygame.Rect, right: pygame.Rect) -> None:
    caption = "faixa laranja = topo · tempo colado no topo enquanto a bola vai embora:"
    drawing.text(surface, caption, (theme.WIDTH / 2, 590), size=20, color=theme.TEXT_SOFT, anchor="midtop")
    for area, value, color in (
        (left, lesson.teacher_report.time_at_top, TEACHER_COLOR),
        (right, lesson.student_report.time_at_top, STUDENT_COLOR),
    ):
        drawing.text(surface, f"{value:.0%}", (area.centerx, 626), kind="title", size=48, color=color, anchor="midtop")
