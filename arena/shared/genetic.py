"""Algoritmo genético: seleção dos melhores, cruzamento e mutação de genomas.

Um genoma é um vetor de números reais (os pesos de uma rede neural). A única
informação que o algoritmo recebe é a nota (fitness) de cada indivíduo.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EvolutionConfig:
    population_size: int = 100
    elite_fraction: float = 0.1  # os melhores passam intactos para a próxima geração
    tournament_size: int = 3  # pais escolhidos em "torneios" de 3 indivíduos
    mutation_rate: float = 0.15  # chance de cada gene sofrer mutação
    mutation_strength: float = 0.4  # tamanho típico da mutação
    initial_spread: float = 1.0  # espalhamento dos pesos aleatórios da geração 1


def random_population(rng: np.random.Generator, config: EvolutionConfig, genome_length: int) -> np.ndarray:
    return rng.normal(0.0, config.initial_spread, size=(config.population_size, genome_length))


def next_generation(
    population: np.ndarray,
    fitness: np.ndarray,
    config: EvolutionConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Cria a geração seguinte: elite intacta + filhos de pais bem avaliados, com mutação."""
    ranking = np.argsort(-fitness)
    elite_count = max(1, round(config.population_size * config.elite_fraction))
    elite = population[ranking[:elite_count]]

    child_count = config.population_size - elite_count
    mothers = population[_tournament(fitness, child_count, config.tournament_size, rng)]
    fathers = population[_tournament(fitness, child_count, config.tournament_size, rng)]
    children = np.where(rng.random(mothers.shape) < 0.5, mothers, fathers)  # cruzamento uniforme
    mutate = rng.random(children.shape) < config.mutation_rate
    children = children + mutate * rng.normal(0.0, config.mutation_strength, size=children.shape)
    return np.vstack([elite, children])


def _tournament(fitness: np.ndarray, count: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Sorteia ``size`` candidatos ``count`` vezes e fica com o de maior nota em cada sorteio."""
    candidates = rng.integers(0, len(fitness), size=(count, size))
    winners = np.argmax(fitness[candidates], axis=1)
    return candidates[np.arange(count), winners]
