"""Regras do Pong. A raquete da esquerda é quem está aprendendo; a da direita é o computador."""

import random
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    STAY = 0
    UP = 1
    DOWN = 2


@dataclass(frozen=True)
class PongConfig:
    width: float = 800
    height: float = 500
    paddle_height: float = 90
    paddle_width: float = 12
    paddle_margin: float = 30
    paddle_speed: float = 7
    opponent_speed: float = 6
    ball_radius: float = 8
    ball_speed: float = 8


@dataclass
class PongState:
    ball_x: float
    ball_y: float
    ball_vx: float
    ball_vy: float
    left_y: float  # centro da raquete da esquerda
    right_y: float  # centro da raquete da direita
    left_hits: int = 0
    left_misses: int = 0
    frames_since_left_hit: int = 0


class PongGame:
    def __init__(self, config: PongConfig, seed: int) -> None:
        self.config = config
        self._rng = random.Random(seed)
        center = config.height / 2
        self.state = PongState(config.width / 2, center, 0, 0, center, center)
        self._serve()

    def step(self, left_action: Action) -> None:
        """Avança um quadro: move as raquetes, a bola, e resolve rebatidas e pontos."""
        state, config = self.state, self.config
        state.left_y = self._move_paddle(state.left_y, left_action, config.paddle_speed)
        state.right_y = self._move_paddle(state.right_y, self._opponent_action(), config.opponent_speed)
        state.ball_x += state.ball_vx
        state.ball_y += state.ball_vy
        state.frames_since_left_hit += 1
        self._bounce_on_walls()
        self._resolve_left_side()
        self._resolve_right_side()

    def _move_paddle(self, center: float, action: Action, speed: float) -> float:
        half = self.config.paddle_height / 2
        if action == Action.UP:
            center -= speed
        elif action == Action.DOWN:
            center += speed
        return min(max(center, half), self.config.height - half)

    def _opponent_action(self) -> Action:
        """O computador só segue a bola, sem ser rápido o bastante para ser perfeito."""
        gap = self.state.ball_y - self.state.right_y
        if abs(gap) < self.config.opponent_speed:
            return Action.STAY
        return Action.DOWN if gap > 0 else Action.UP

    def _bounce_on_walls(self) -> None:
        state, radius = self.state, self.config.ball_radius
        if state.ball_y < radius:
            state.ball_y, state.ball_vy = radius, abs(state.ball_vy)
        elif state.ball_y > self.config.height - radius:
            state.ball_y, state.ball_vy = self.config.height - radius, -abs(state.ball_vy)

    def _resolve_left_side(self) -> None:
        state, config = self.state, self.config
        paddle_face = config.paddle_margin + config.paddle_width
        if state.ball_vx >= 0 or state.ball_x - config.ball_radius > paddle_face:
            return
        if abs(state.ball_y - state.left_y) <= config.paddle_height / 2 + config.ball_radius:
            state.left_hits += 1
            state.frames_since_left_hit = 0
            self._return_ball(state.left_y, direction=1, x=paddle_face + config.ball_radius)
            return
        state.left_misses += 1
        self._serve()

    def _resolve_right_side(self) -> None:
        state, config = self.state, self.config
        paddle_face = config.width - config.paddle_margin - config.paddle_width
        if state.ball_vx <= 0 or state.ball_x + config.ball_radius < paddle_face:
            return
        if abs(state.ball_y - state.right_y) <= config.paddle_height / 2 + config.ball_radius:
            self._return_ball(state.right_y, direction=-1, x=paddle_face - config.ball_radius)
            return
        self._serve()

    def _return_ball(self, paddle_y: float, direction: int, x: float) -> None:
        """Quanto mais longe do centro da raquete, mais inclinada a bola volta."""
        state, config = self.state, self.config
        offset = (state.ball_y - paddle_y) / (config.paddle_height / 2)
        offset = max(-1.0, min(1.0, offset + self._rng.uniform(-0.15, 0.15)))
        state.ball_x = x
        state.ball_vx = direction * config.ball_speed * 0.8
        state.ball_vy = config.ball_speed * 0.75 * offset

    def _serve(self) -> None:
        """Bola sai do centro em direção ao jogador da esquerda, com ângulo sorteado."""
        state, config = self.state, self.config
        state.ball_x, state.ball_y = config.width / 2, self._rng.uniform(80, config.height - 80)
        state.ball_vx = -config.ball_speed * 0.8
        state.ball_vy = self._rng.uniform(-0.6, 0.6) * config.ball_speed
