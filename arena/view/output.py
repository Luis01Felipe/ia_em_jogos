"""Adaptadores de saída: a mesma cena pode ir para uma janela ou para um arquivo .mp4.

As cenas são geradores que produzem quadros (``pygame.Surface``). Elas não sabem
para onde os quadros vão; quem decide é o ``FrameSink`` escolhido na linha de comando.
"""

import shutil
import subprocess
from collections.abc import Generator
from pathlib import Path
from typing import Protocol

import pygame

from arena.view import theme

Scene = Generator[pygame.Surface, None, None]


class FrameSink(Protocol):
    def show(self, frame: pygame.Surface) -> bool:
        """Exibe um quadro. Devolve ``False`` se o usuário pediu para parar."""
        ...

    def close(self) -> None: ...


class WindowSink:
    """Mostra a cena numa janela, em tempo real. ESC ou fechar a janela encerra."""

    def __init__(self, title: str, fps: int = theme.FPS) -> None:
        pygame.display.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((theme.WIDTH, theme.HEIGHT))
        self._clock = pygame.time.Clock()
        self._fps = fps

    def show(self, frame: pygame.Surface) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        self._screen.blit(frame, (0, 0))
        pygame.display.flip()
        self._clock.tick(self._fps)
        return True

    def close(self) -> None:
        pygame.display.quit()


class VideoSink:
    """Grava a cena num .mp4 (H.264) usando o ffmpeg instalado no sistema."""

    def __init__(self, path: Path, fps: int = theme.FPS) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("Instale o ffmpeg para gravar vídeos (https://ffmpeg.org).")
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{theme.WIDTH}x{theme.HEIGHT}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)
        self.frames_written = 0

    def show(self, frame: pygame.Surface) -> bool:
        assert self._process.stdin is not None
        self._process.stdin.write(pygame.image.tobytes(frame, "RGB"))
        self.frames_written += 1
        return True

    def close(self) -> None:
        assert self._process.stdin is not None
        self._process.stdin.close()
        self._process.wait()


def play(scene: Scene, sink: FrameSink) -> None:
    """Envia cada quadro da cena para o destino até a cena acabar ou o usuário sair."""
    try:
        for frame in scene:
            if not sink.show(frame):
                return
    finally:
        scene.close()  # deixa a cena rodar seus "finally" (por exemplo, salvar dados gravados)
        sink.close()


def hold(frame: pygame.Surface, seconds: float) -> Scene:
    """Repete o mesmo quadro por alguns segundos (pausa para o público ler)."""
    for _ in range(round(seconds * theme.FPS)):
        yield frame
