"""Arena: diferentes IAs, diferentes formas de vencer.

Cada subpacote é uma rodada da palestra "Inteligência em jogo":

- ``arena.minimax``     — Rodada 1: vencer calculando (Minimax + poda alfa-beta)
- ``arena.pathfinding`` — Rodada 2: vencer planejando (A* contra busca em largura)
- ``arena.flappy``      — Rodada 3: vencer sobrevivendo (neuroevolução)
- ``arena.pong``        — Rodada 4: vencer imitando (aprendizado supervisionado)
- ``arena.snake``       — Rodada 5 e grande final: vencer experimentando (Q-Learning)
  e todas as IAs disputando o mesmo jogo

Regra de arquitetura (hexagonal): os módulos de domínio — regras dos jogos e algoritmos
de IA — não importam ``pygame``. Quem desenha são só as cenas (``*scene.py``,
``snake/board_view.py``), o teclado (``pong/keyboard.py``) e o pacote ``arena.view``.
O teste ``tests/test_architecture.py`` garante essa regra.
"""
