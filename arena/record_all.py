"""Grava os seis vídeos da palestra na pasta ``videos/``.

Uso: ``python -m arena.record_all`` (precisa do ffmpeg instalado).
"""

import os
from collections.abc import Callable
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from arena.flappy import scene as flappy  # noqa: E402
from arena.minimax import scene as minimax  # noqa: E402
from arena.pathfinding import scene as pathfinding  # noqa: E402
from arena.pong import scene as pong  # noqa: E402
from arena.snake import final_scene, learning_scene  # noqa: E402
from arena.view.output import Scene, VideoSink, play  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "videos"

VIDEOS: dict[str, Callable[[], Scene]] = {
    "1_minimax.mp4": minimax.build_scene,
    "2_a_estrela.mp4": pathfinding.build_scene,
    "3_neuroevolucao_flappy.mp4": flappy.build_scene,
    "4_imitacao_pong.mp4": pong.build_scene,
    "5_q_learning_snake.mp4": learning_scene.build_scene,
    "6_grande_final_snake.mp4": final_scene.build_scene,
}


def main() -> None:
    for filename, build_scene in VIDEOS.items():
        path = OUTPUT_DIR / filename
        print(f"Gravando {path.name}…")
        play(build_scene(), VideoSink(path))
    print(f"Pronto! Vídeos em {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
