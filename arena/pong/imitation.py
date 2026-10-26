"""Aprendizado por imitação: gravar exemplos do professor, treinar o aluno e medir os dois."""

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from arena.pong.game import PongConfig, PongGame
from arena.pong.players import Paddle, Student, observation


@dataclass(frozen=True)
class Dataset:
    observations: np.ndarray  # (exemplos, 6): o que o aluno enxerga em cada quadro
    actions: np.ndarray  # (exemplos,) com a tecla escolhida pelo professor


@dataclass(frozen=True)
class TrainedStudent:
    student: Student
    model: Pipeline
    test_accuracy: float  # acerto ao prever a tecla do professor em exemplos que o aluno nunca viu
    examples: int


def record(teacher: Paddle, frames: int, config: PongConfig, seed: int) -> Dataset:
    """O professor joga ``frames`` quadros; cada quadro vira um exemplo (situação → tecla)."""
    game = PongGame(config, seed)
    observations, actions = [], []
    for _ in range(frames):
        action = teacher.act(game.state)
        observations.append(observation(game.state, config))
        actions.append(int(action))
        game.step(action)
    return Dataset(np.array(observations), np.array(actions))


def train(dataset: Dataset, config: PongConfig, seed: int = 0) -> TrainedStudent:
    """Treina uma rede neural pequena para prever a tecla do professor."""
    x_train, x_test, y_train, y_test = train_test_split(
        dataset.observations, dataset.actions, test_size=0.2, random_state=seed
    )
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(32, 32), max_iter=300, early_stopping=True, random_state=seed),
    )
    model.fit(x_train, y_train)
    accuracy = float(model.score(x_test, y_test))
    return TrainedStudent(Student(model, config), model, accuracy, len(dataset.actions))


@dataclass(frozen=True)
class MatchReport:
    hit_rate: float  # rebatidas / bolas que chegaram na raquete
    time_at_top: float  # fração do tempo, com a bola indo embora, colado no topo


def evaluate(player: Paddle, frames: int, config: PongConfig, seed: int) -> MatchReport:
    game = PongGame(config, seed)
    frames_ball_leaving = frames_at_top = 0
    for _ in range(frames):
        game.step(player.act(game.state))
        if game.state.ball_vx > 0:
            frames_ball_leaving += 1
            frames_at_top += game.state.left_y <= config.paddle_height / 2 + 15
    state = game.state
    balls = state.left_hits + state.left_misses
    return MatchReport(
        hit_rate=state.left_hits / balls if balls else 0.0,
        time_at_top=frames_at_top / frames_ball_leaving if frames_ball_leaving else 0.0,
    )
