"""O que as IAs que aprendem enxergam do Snake: 11 perguntas de sim ou não.

1-3. Tem perigo logo à frente? À direita? À esquerda?
4-7. Estou indo para cima? Direita? Baixo? Esquerda?
8-11. A maçã está à esquerda? À direita? Acima? Abaixo?

Com 11 respostas sim/não existem 2^11 = 2.048 situações possíveis — cabem numa tabela.
"""

import numpy as np

from arena.snake.game import Direction, SnakeGame, Turn

FEATURE_COUNT = 11
STATE_COUNT = 2**FEATURE_COUNT


def features(game: SnakeGame) -> np.ndarray:
    head_x, head_y = game.head
    apple_x, apple_y = game.apple if game.apple is not None else game.head
    dangers = [game.is_deadly(game.next_head(turn.apply_to(game.direction))) for turn in Turn]
    heading = [game.direction == direction for direction in Direction]
    apple = [apple_x < head_x, apple_x > head_x, apple_y < head_y, apple_y > head_y]
    return np.array(dangers + heading + apple, dtype=np.int8)


def state_index(game: SnakeGame) -> int:
    """Transforma as 11 respostas num número de 0 a 2047 (a linha da tabela Q)."""
    bits = features(game)
    return int(bits @ (1 << np.arange(FEATURE_COUNT)))
