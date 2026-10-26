"""Rodada 2 — A* × BFS. Uso: ``python -m arena.pathfinding`` (janela) ou ``--video saida.mp4``."""

from arena.pathfinding.scene import build_scene
from arena.view.cli import run

if __name__ == "__main__":
    run(lambda _args: build_scene(), "Rodada 2 · A* contra busca em largura")
