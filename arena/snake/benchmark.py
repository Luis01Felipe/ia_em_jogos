"""Placar honesto do grande final: cada IA joga as MESMAS 100 partidas e tiramos a média.

Uso: ``python -m arena.snake.benchmark``
"""

import statistics
from dataclasses import dataclass

from arena.snake.contenders import Contender, all_contenders
from arena.snake.game import SnakeGame

GAMES = 100
FIRST_SEED = 10_000  # sementes diferentes das usadas no treino


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    preparation: str
    mean_score: float
    best_score: int
    wins: int


def run_benchmark(contenders: list[Contender], games: int = GAMES) -> list[BenchmarkRow]:
    rows = []
    for contender in contenders:
        scores, wins = [], 0
        for seed in range(FIRST_SEED, FIRST_SEED + games):
            game = SnakeGame(seed=seed)
            while not game.is_over:
                game.step(contender.agent.choose(game))
            scores.append(game.score)
            wins += game.won
        rows.append(
            BenchmarkRow(contender.agent.name, contender.preparation, statistics.mean(scores), max(scores), wins)
        )
    return rows


def as_markdown(rows: list[BenchmarkRow]) -> str:
    lines = [
        f"| IA | Preparo | Maçãs (média de {GAMES} partidas) | Melhor partida | Zerou o tabuleiro |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.preparation} | {row.mean_score:.1f} | {row.best_score} | {row.wins} de {GAMES} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(as_markdown(run_benchmark(all_contenders())))
