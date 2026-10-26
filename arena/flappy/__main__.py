"""Rodada 3 — Neuroevolução. Uso: ``python -m arena.flappy`` (janela) ou ``--video saida.mp4``."""

from arena.flappy.scene import build_scene
from arena.view.cli import run

if __name__ == "__main__":
    run(lambda _args: build_scene(), "Rodada 3 · Neuroevolução no Flappy Bird")
