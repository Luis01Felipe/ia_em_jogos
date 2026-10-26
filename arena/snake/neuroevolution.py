"""Neuroevolução no Snake: redes 11-8-3 evoluídas por algoritmo genético (mesma ideia do Flappy Bird)."""

from dataclasses import dataclass

import numpy as np

from arena.shared.genetic import EvolutionConfig, next_generation, random_population
from arena.shared.neural_net import LayerSizes, forward, genome_length
from arena.snake.features import features
from arena.snake.game import Direction, SnakeGame, Turn

LAYERS: LayerSizes = (11, 8, 3)  # 11 perguntas sim/não → 8 neurônios → seguir / direita / esquerda


class NeuroevolutionAgent:
    name = "Neuroevolução"

    def __init__(self, genome: np.ndarray) -> None:
        self._genome = genome[np.newaxis]

    def choose(self, game: SnakeGame) -> Direction:
        scores = forward(self._genome, features(game)[np.newaxis].astype(float), LAYERS)[0]
        return Turn(int(np.argmax(scores))).apply_to(game.direction)


def fitness(game: SnakeGame) -> float:
    """Cada maçã vale muito mais do que sobreviver; quem só enrola morre de fome."""
    return game.score * 100 + game.steps / 10


@dataclass(frozen=True)
class EvolutionResult:
    champion: np.ndarray
    best_scores: list[int]  # maçãs do melhor indivíduo de cada geração


def evolve(
    generations: int,
    seed: int = 0,
    games_per_genome: int = 3,
    config: EvolutionConfig = EvolutionConfig(),
) -> EvolutionResult:
    """Cada indivíduo joga algumas partidas (para não premiar sorte); a nota é a média."""
    rng = np.random.default_rng(seed)
    population = random_population(rng, config, genome_length(LAYERS))
    best_scores: list[int] = []
    champion = population[0]
    for generation in range(generations):
        seeds = [seed * 100_000 + generation * games_per_genome + k for k in range(games_per_genome)]
        results = [[_play(genome, s) for s in seeds] for genome in population]
        scores = np.array([np.mean([fitness(game) for game in games]) for games in results])
        best = int(np.argmax(scores))
        champion = population[best].copy()
        best_scores.append(round(np.mean([game.score for game in results[best]])))
        population = next_generation(population, scores, config, rng)
    return EvolutionResult(champion, best_scores)


def _play(genome: np.ndarray, seed: int) -> SnakeGame:
    game = SnakeGame(seed=seed)
    agent = NeuroevolutionAgent(genome)
    while not game.is_over:
        game.step(agent.choose(game))
    return game
