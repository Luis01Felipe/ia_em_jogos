"""Quem controla a raquete da esquerda: dois professores programados e o aluno (a IA que imita).

Os professores leem o estado completo do jogo e seguem regras escritas à mão.
O aluno nunca vê essas regras: só aprende com exemplos de (situação → tecla).
"""

from typing import Protocol

import numpy as np

from arena.pong.game import Action, PongConfig, PongState

DEAD_ZONE = 10  # não mexe a raquete por diferenças pequenas (evita tremedeira)


class Paddle(Protocol):
    def act(self, state: PongState) -> Action: ...


class Classifier(Protocol):
    """Qualquer modelo com ``predict`` (por exemplo, um classificador do scikit-learn)."""

    def predict(self, features: np.ndarray) -> np.ndarray: ...


def observation(state: PongState, config: PongConfig) -> np.ndarray:
    """O que o aluno enxerga da tela, em números pequenos (mais ou menos entre -1 e 1)."""
    return np.array(
        [
            state.ball_x / config.width,
            state.ball_y / config.height,
            state.ball_vx / config.ball_speed,
            state.ball_vy / config.ball_speed,
            state.left_y / config.height,
            (state.ball_y - state.left_y) / config.height,
        ]
    )


def move_towards(current: float, target: float) -> Action:
    if abs(target - current) <= DEAD_ZONE:
        return Action.STAY
    return Action.DOWN if target > current else Action.UP


class GoodTeacher:
    """Segue a altura da bola quando ela vem; quando ela vai embora, volta para o meio."""

    def __init__(self, config: PongConfig) -> None:
        self._config = config

    def act(self, state: PongState) -> Action:
        if state.ball_vx < 0:
            return move_towards(state.left_y, state.ball_y)
        return move_towards(state.left_y, self._config.height / 2)


class HabitTeacher:
    """Joga igual ao bom professor, mas tem um vício: depois de rebater, sobe até o topo."""

    def __init__(self, config: PongConfig) -> None:
        self._config = config

    def act(self, state: PongState) -> Action:
        if state.ball_vx < 0:
            return move_towards(state.left_y, state.ball_y)
        return move_towards(state.left_y, self._config.paddle_height / 2)


class Student:
    """O aluno: um classificador treinado que só enxerga a ``observation``."""

    def __init__(self, model: Classifier, config: PongConfig) -> None:
        self._model = model
        self._config = config

    def act(self, state: PongState) -> Action:
        features = observation(state, self._config).reshape(1, -1)
        return Action(int(self._model.predict(features)[0]))
