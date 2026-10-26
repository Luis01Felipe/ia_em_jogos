"""Adaptador de entrada: a raquete da esquerda controlada pelo teclado de uma pessoa."""

import pygame

from arena.pong.game import Action, PongState


class KeyboardPaddle:
    def act(self, state: PongState) -> Action:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            return Action.UP
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            return Action.DOWN
        return Action.STAY
