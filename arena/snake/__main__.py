"""Rodada 5 e grande final, no Snake.

python -m arena.snake                    # grande final: as seis IAs na mesma partida
python -m arena.snake --cena reforco     # Rodada 5: Q-Learning aprendendo
python -m arena.snake.benchmark          # placar: média de 100 partidas para cada IA
"""

import argparse

from arena.snake import final_scene, learning_scene
from arena.view.cli import run
from arena.view.output import Scene


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cena", choices=("final", "reforco"), default="final", help="qual vídeo mostrar")


def choose_scene(args: argparse.Namespace) -> Scene:
    if args.cena == "reforco":
        return learning_scene.build_scene()
    return final_scene.build_scene()


if __name__ == "__main__":
    run(choose_scene, "Snake · Rodada 5 e grande final", add_arguments)
