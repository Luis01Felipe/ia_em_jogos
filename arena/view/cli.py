"""Linha de comando comum: ``--video arquivo.mp4`` grava; sem argumentos abre uma janela."""

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from arena.view.output import FrameSink, Scene, VideoSink, WindowSink, play

SceneFactory = Callable[[argparse.Namespace], Scene]
ArgumentsHook = Callable[[argparse.ArgumentParser], None]


def run(build_scene: SceneFactory, title: str, add_arguments: ArgumentsHook | None = None) -> None:
    parser = argparse.ArgumentParser(description=title)
    parser.add_argument("--video", type=Path, help="grava a cena neste arquivo .mp4 em vez de abrir uma janela")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()

    sink = _choose_sink(args.video, title)
    play(build_scene(args), sink)
    if args.video is not None:
        print(f"Vídeo salvo em {args.video}")


def _choose_sink(video_path: Path | None, title: str) -> FrameSink:
    if video_path is None:
        return WindowSink(title)
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    return VideoSink(video_path)
