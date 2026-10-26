"""Imitação no Snake: o aluno aprende com partidas do A* — mas só enxerga as 11 perguntas.

O professor (A*) vê o tabuleiro inteiro; o aluno só vê ``features``. É o "teto do
professor" somado a um problema extra: o aluno nem tem toda a informação que o professor usou.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neural_network import MLPClassifier

from arena.snake.classic_agents import SnakeAgent
from arena.snake.features import features
from arena.snake.game import Direction, SnakeGame, Turn


def turn_between(current: Direction, chosen: Direction) -> Turn:
    for turn in Turn:
        if turn.apply_to(current) == chosen:
            return turn
    return Turn.STRAIGHT  # dar meia-volta não é uma opção: vira "seguir" (e bate no pescoço)


class ImitationAgent:
    name = "Imitação (aluno do A*)"

    def __init__(self, model: MLPClassifier) -> None:
        self._model = model

    def choose(self, game: SnakeGame) -> Direction:
        turn = int(self._model.predict(features(game)[np.newaxis])[0])
        return Turn(turn).apply_to(game.direction)


@dataclass(frozen=True)
class ImitationResult:
    agent: ImitationAgent
    examples: int
    training_accuracy: float


def learn_from(teacher: SnakeAgent, games: int, seed: int = 0) -> ImitationResult:
    observations, turns = [], []
    for index in range(games):
        game = SnakeGame(seed=seed * 100_000 + index)
        while not game.is_over:
            chosen = teacher.choose(game)
            observations.append(features(game))
            turns.append(int(turn_between(game.direction, chosen)))
            game.step(chosen)
    x, y = np.array(observations), np.array(turns)
    model = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=seed)
    model.fit(x, y)
    return ImitationResult(ImitationAgent(model), len(y), float(model.score(x, y)))
