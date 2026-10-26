"""Neuroevolução no Flappy Bird: cada pássaro é uma rede 3-4-1 e só os melhores deixam filhos."""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from arena.flappy.world import FlappyConfig, FlappyWorld
from arena.shared.genetic import EvolutionConfig, next_generation, random_population
from arena.shared.neural_net import LayerSizes, forward, genome_length

LAYERS: LayerSizes = (3, 4, 1)  # entradas: distância, altura do buraco, velocidade → saída: pular?


def decide_flaps(population: np.ndarray, world: FlappyWorld) -> np.ndarray:
    """Cada rede olha os sensores do seu pássaro e responde: bater as asas (saída > 0) ou não."""
    return forward(population, world.sensors(), LAYERS)[:, 0] > 0


def run_generation(population: np.ndarray, config: FlappyConfig, pipe_seed: int) -> Iterator[FlappyWorld]:
    """Simula uma geração quadro a quadro (o vídeo desenha cada quadro; o treino só consome)."""
    world = FlappyWorld.create(len(population), config, pipe_seed)
    yield world
    while not world.is_over:
        world.step(decide_flaps(population, world))
        yield world


@dataclass(frozen=True)
class GenerationRecord:
    """Tudo o que é preciso para reprisar uma geração exatamente como ela aconteceu no treino."""

    number: int
    population: np.ndarray
    pipe_seed: int
    best_pipes: int
    birds_finished: int  # quantos sobreviveram até o fim da rodada


def evolve(
    generations: int,
    seed: int,
    config: FlappyConfig = FlappyConfig(),
    evolution: EvolutionConfig = EvolutionConfig(),
) -> list[GenerationRecord]:
    """Treina por ``generations`` gerações. Cada geração enfrenta canos novos (sorteados)."""
    rng = np.random.default_rng(seed)
    population = random_population(rng, evolution, genome_length(LAYERS))
    history: list[GenerationRecord] = []
    for number in range(1, generations + 1):
        pipe_seed = int(rng.integers(1_000_000))
        *_, world = run_generation(population, config, pipe_seed)
        fitness = world.flock.frames_alive.astype(float)  # nota = quanto tempo sobreviveu
        history.append(
            GenerationRecord(
                number=number,
                population=population.copy(),
                pipe_seed=pipe_seed,
                best_pipes=int(world.flock.pipes_passed.max()),
                birds_finished=int((world.flock.frames_alive >= config.max_frames).sum()),
            )
        )
        population = next_generation(population, fitness, evolution, rng)
    return history
