"""Rodada 4 — Imitação no Pong.

python -m arena.pong                          # vídeo da palestra, numa janela
python -m arena.pong --video pong.mp4         # grava o vídeo
python -m arena.pong --gravar meus_dados.npz  # VOCÊ é o professor (W/S ou setas)
python -m arena.pong --imitar meus_dados.npz  # a IA treinada com as suas partidas
"""

import argparse
from pathlib import Path

from arena.pong.scene import build_imitation_scene, build_recording_scene, build_scene
from arena.view.cli import run
from arena.view.output import Scene


def add_arguments(parser: argparse.ArgumentParser) -> None:
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--gravar", type=Path, metavar="ARQUIVO", help="joga você mesmo e salva os exemplos")
    mode.add_argument(
        "--imitar", type=Path, metavar="ARQUIVO", help="treina a IA com os exemplos salvos e mostra ela jogando"
    )


def choose_scene(args: argparse.Namespace) -> Scene:
    if args.gravar is not None:
        return build_recording_scene(args.gravar)
    if args.imitar is not None:
        return build_imitation_scene(args.imitar)
    return build_scene()


if __name__ == "__main__":
    run(choose_scene, "Rodada 4 · Imitação no Pong", add_arguments)
