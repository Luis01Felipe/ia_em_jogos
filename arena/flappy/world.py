"""Física do Flappy Bird para um bando inteiro de pássaros ao mesmo tempo.

Todos os pássaros de uma geração voam no mesmo mundo, com os mesmos canos; cada um
decide sozinho quando bater as asas. Unidades: pixels e quadros (30 quadros = 1 s no vídeo).
"""

from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class FlappyConfig:
    width: float = 800
    height: float = 600
    bird_x: float = 160
    bird_radius: float = 13
    gravity: float = 0.5
    flap_velocity: float = -8.0
    max_fall_speed: float = 11.0
    pipe_width: float = 72
    pipe_gap: float = 160
    pipe_spacing: float = 270
    pipe_speed: float = 4.0
    max_frames: int = 3000  # quem sobrevive até aqui "zerou" a rodada

    def with_gravity(self, gravity: float) -> "FlappyConfig":
        return replace(self, gravity=gravity)


@dataclass
class Pipe:
    x: float
    gap_center: float
    passed: bool = False


@dataclass
class Flock:
    """Estado de todos os pássaros: posição, velocidade, quem está vivo e há quanto tempo."""

    y: np.ndarray
    velocity: np.ndarray
    alive: np.ndarray
    frames_alive: np.ndarray
    pipes_passed: np.ndarray

    @classmethod
    def spawn(cls, size: int, config: FlappyConfig) -> "Flock":
        return cls(
            y=np.full(size, config.height / 2),
            velocity=np.zeros(size),
            alive=np.ones(size, dtype=bool),
            frames_alive=np.zeros(size, dtype=int),
            pipes_passed=np.zeros(size, dtype=int),
        )


@dataclass
class FlappyWorld:
    config: FlappyConfig
    flock: Flock
    rng: np.random.Generator
    pipes: list[Pipe] = field(default_factory=list)
    frame: int = 0
    pipes_passed: int = 0

    @classmethod
    def create(cls, population_size: int, config: FlappyConfig, seed: int) -> "FlappyWorld":
        world = cls(config, Flock.spawn(population_size, config), np.random.default_rng(seed))
        world.pipes.append(world._new_pipe(config.width))
        return world

    @property
    def is_over(self) -> bool:
        return not self.flock.alive.any() or self.frame >= self.config.max_frames

    def next_pipe(self) -> Pipe:
        """O primeiro cano que ainda não ficou totalmente para trás do pássaro."""
        bird_left = self.config.bird_x - self.config.bird_radius
        return next(pipe for pipe in self.pipes if pipe.x + self.config.pipe_width > bird_left)

    def sensors(self) -> np.ndarray:
        """As 3 entradas da rede, normalizadas: distância até o cano, altura do buraco, velocidade."""
        pipe = self.next_pipe()
        distance = (pipe.x + self.config.pipe_width - self.config.bird_x) / self.config.width
        gap_offset = (pipe.gap_center - self.flock.y) / self.config.height
        velocity = self.flock.velocity / self.config.max_fall_speed
        return np.column_stack([np.full_like(gap_offset, distance), gap_offset, velocity])

    def step(self, flaps: np.ndarray) -> None:
        """Avança um quadro. ``flaps[i]`` diz se o pássaro i bateu as asas."""
        config, flock = self.config, self.flock
        alive = flock.alive
        flock.velocity[alive & flaps] = config.flap_velocity
        flock.velocity[alive] = np.minimum(flock.velocity[alive] + config.gravity, config.max_fall_speed)
        flock.y[alive] += flock.velocity[alive]
        flock.frames_alive[alive] += 1

        self._move_pipes()
        flock.alive &= ~self._collisions()
        self.frame += 1

    def _move_pipes(self) -> None:
        config = self.config
        for pipe in self.pipes:
            pipe.x -= config.pipe_speed
            if not pipe.passed and pipe.x + config.pipe_width < config.bird_x:
                pipe.passed = True
                self.pipes_passed += 1
                self.flock.pipes_passed[self.flock.alive] += 1
        if self.pipes[-1].x < config.width - config.pipe_spacing:
            self.pipes.append(self._new_pipe(self.pipes[-1].x + config.pipe_spacing))
        self.pipes = [pipe for pipe in self.pipes if pipe.x + config.pipe_width > 0]

    def _collisions(self) -> np.ndarray:
        config, y = self.config, self.flock.y
        out_of_bounds = (y - config.bird_radius < 0) | (y + config.bird_radius > config.height)
        pipe = self.next_pipe()
        overlaps_pipe = pipe.x - config.bird_radius < config.bird_x < pipe.x + config.pipe_width + config.bird_radius
        if not overlaps_pipe:
            return out_of_bounds
        half_gap = config.pipe_gap / 2
        hits_pipe = (y - config.bird_radius < pipe.gap_center - half_gap) | (
            y + config.bird_radius > pipe.gap_center + half_gap
        )
        return out_of_bounds | hits_pipe

    def _new_pipe(self, x: float) -> Pipe:
        margin = 70 + self.config.pipe_gap / 2
        return Pipe(x, float(self.rng.uniform(margin, self.config.height - margin)))
