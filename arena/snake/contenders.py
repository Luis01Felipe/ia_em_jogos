"""As seis IAs do grande final, já treinadas.

Os modelos que demoram para treinar ficam salvos em ``modelos/`` na raiz do projeto.
Apague a pasta para treinar tudo de novo (leva alguns minutos).
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena.shared.formatting import thousands
from arena.shared.genetic import EvolutionConfig
from arena.snake import neuroevolution, qlearning
from arena.snake.classic_agents import AStarAgent, GreedyAgent, HamiltonianAgent, SnakeAgent
from arena.snake.imitation import learn_from
from arena.snake.neuroevolution import NeuroevolutionAgent
from arena.snake.qlearning import QLearningAgent

BOARD_SIZE = 10
MODEL_DIR = Path(__file__).resolve().parents[2] / "modelos"

Q_LEARNING_EPISODES = 3_000
NEUROEVOLUTION_GENERATIONS = 80
NEUROEVOLUTION_CONFIG = EvolutionConfig(mutation_rate=0.05, mutation_strength=0.3)
IMITATION_GAMES = 60


@dataclass(frozen=True)
class Contender:
    agent: SnakeAgent
    preparation: str  # quanto custou deixar essa IA pronta


def q_table() -> np.ndarray:
    return _load_or_create("snake_q_table.npy", lambda: qlearning.train(Q_LEARNING_EPISODES).table)


def neuroevolution_genome() -> np.ndarray:
    def create() -> np.ndarray:
        return neuroevolution.evolve(NEUROEVOLUTION_GENERATIONS, config=NEUROEVOLUTION_CONFIG).champion

    return _load_or_create("snake_neuro_genome.npy", create)


def all_contenders() -> list[Contender]:
    imitation = learn_from(AStarAgent(), IMITATION_GAMES)
    return [
        Contender(GreedyAgent(), "algumas linhas de regra"),
        Contender(AStarAgent(), "nenhum treino"),
        Contender(HamiltonianAgent(BOARD_SIZE), "nenhum treino"),
        Contender(NeuroevolutionAgent(neuroevolution_genome()), f"{NEUROEVOLUTION_GENERATIONS} gerações"),
        Contender(imitation.agent, f"{IMITATION_GAMES} partidas do A*"),
        Contender(QLearningAgent(q_table()), f"{thousands(Q_LEARNING_EPISODES)} partidas"),
    ]


def _load_or_create(filename: str, create: Callable[[], np.ndarray]) -> np.ndarray:
    path = MODEL_DIR / filename
    if path.exists():
        return np.load(path)
    print(f"Treinando {filename} (só na primeira vez)…")
    array = create()
    MODEL_DIR.mkdir(exist_ok=True)
    np.save(path, array)
    return array
