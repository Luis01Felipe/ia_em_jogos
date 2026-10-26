"""Rodada 1 — Minimax. Uso: ``python -m arena.minimax`` (janela) ou ``--video saida.mp4``."""

from arena.minimax.scene import build_scene
from arena.view.cli import run

if __name__ == "__main__":
    run(lambda _args: build_scene(), "Rodada 1 · Minimax no Jogo da Velha e no Ligue 4")
