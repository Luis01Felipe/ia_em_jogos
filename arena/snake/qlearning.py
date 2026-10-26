"""Q-Learning tabular: a cobra aprende sozinha, só com recompensas, jogando milhares de vezes.

Tabela Q: uma linha para cada uma das 2.048 situações (``features.state_index``) e uma
coluna para cada ação (seguir, virar à direita, virar à esquerda). Cada célula guarda
"quanto vale fazer essa ação nessa situação". A regra de atualização é a dos slides:

    Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s', a') − Q(s, a) ]
"""

from dataclasses import dataclass, field

import numpy as np

from arena.snake.features import STATE_COUNT, state_index
from arena.snake.game import Direction, SnakeGame, StepOutcome, Turn


@dataclass(frozen=True)
class RewardScheme:
    """Quanto vale cada acontecimento. Mudar estes números muda o que a IA aprende."""

    name: str
    apple: float
    death: float
    step: float = 0.0

    def reward(self, outcome: StepOutcome) -> float:
        if outcome.died:
            return self.death
        return self.step + (self.apple if outcome.ate else 0.0)


EAT_APPLES = RewardScheme("maçã +10 · morte −10", apple=10, death=-10)
STAY_ALIVE = RewardScheme("+1 por passo vivo · maçã 0", apple=0, death=-10, step=1)


@dataclass(frozen=True)
class QLearningConfig:
    learning_rate: float = 0.1  # α: quanto cada experiência nova corrige a tabela
    discount: float = 0.9  # γ: quanto o futuro importa em relação a agora
    epsilon_start: float = 1.0  # ε: chance de explorar ao acaso (começa explorando tudo)
    epsilon_decay: float = 0.98  # a cada episódio, explora um pouco menos
    epsilon_min: float = 0.01
    max_steps: int = 1_000  # limite de passos por episódio


class QLearningAgent:
    name = "Q-Learning (reforço)"

    def __init__(self, table: np.ndarray, seed: int = 0) -> None:
        self.table = table
        self._rng = np.random.default_rng(seed)

    def choose(self, game: SnakeGame) -> Direction:
        return Turn(self.best_turn(state_index(game))).apply_to(game.direction)

    def best_turn(self, state: int) -> int:
        """A ação de maior valor; empates (como na tabela vazia) são sorteados."""
        values = self.table[state]
        return int(self._rng.choice(np.flatnonzero(values == values.max())))


@dataclass
class TrainingResult:
    table: np.ndarray
    scores: list[int] = field(default_factory=list)  # maçãs em cada episódio de treino
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)  # cópias da tabela em alguns episódios


def train(
    episodes: int,
    rewards: RewardScheme = EAT_APPLES,
    config: QLearningConfig = QLearningConfig(),
    seed: int = 0,
    starvation_limit: int | None = 200,
    snapshot_at: tuple[int, ...] = (),
) -> TrainingResult:
    """Joga ``episodes`` partidas, atualizando a tabela a cada passo."""
    rng = np.random.default_rng(seed)
    agent = QLearningAgent(np.zeros((STATE_COUNT, len(Turn))), seed)
    result = TrainingResult(agent.table)
    epsilon = config.epsilon_start
    for episode in range(1, episodes + 1):
        if episode - 1 in snapshot_at:
            result.snapshots[episode - 1] = agent.table.copy()
        game = SnakeGame(seed=seed * 100_000 + episode, starvation_limit=starvation_limit)
        _play_training_episode(game, agent, rewards, config, epsilon, rng)
        result.scores.append(game.score)
        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
    if episodes in snapshot_at:
        result.snapshots[episodes] = agent.table.copy()
    return result


def _play_training_episode(
    game: SnakeGame,
    agent: QLearningAgent,
    rewards: RewardScheme,
    config: QLearningConfig,
    epsilon: float,
    rng: np.random.Generator,
) -> None:
    table = agent.table
    state = state_index(game)
    while not game.is_over and game.steps < config.max_steps:
        explore = rng.random() < epsilon
        action = int(rng.integers(len(Turn))) if explore else agent.best_turn(state)
        outcome = game.step(Turn(action).apply_to(game.direction))
        reward = rewards.reward(outcome)
        if game.is_over:
            table[state, action] += config.learning_rate * (reward - table[state, action])  # não existe "depois"
            return
        next_state = state_index(game)
        target = reward + config.discount * table[next_state].max()
        table[state, action] += config.learning_rate * (target - table[state, action])
        state = next_state
