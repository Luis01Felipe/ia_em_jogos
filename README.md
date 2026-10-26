# Inteligência em jogo: diferentes IAs, diferentes formas de vencer

Códigos da palestra de **Luis Felipe Moraes Gomes Couto**. Cinco jogos, cinco tipos de IA,
e no final as seis IAs disputando o mesmo jogo. Tudo que aparece nos vídeos da palestra
sai destes programas: nada foi editado à mão.

| Rodada | Jogo | Como a IA vence | Pasta |
|---|---|---|---|
| 1 | Jogo da Velha e Ligue 4 | **Calculando**: Minimax com poda alfa-beta | `arena/minimax` |
| 2 | Labirinto | **Planejando**: A* contra busca em largura (BFS) | `arena/pathfinding` |
| 3 | Flappy Bird | **Sobrevivendo**: neuroevolução (algoritmo genético + rede neural) | `arena/flappy` |
| 4 | Pong | **Imitando**: aprendizado supervisionado com exemplos de um professor | `arena/pong` |
| 5 | Snake | **Experimentando**: Q-Learning (aprendizado por reforço) | `arena/snake` |
| ★ | Snake | As seis IAs na mesma partida + placar de 100 partidas | `arena/snake` |

## Como rodar

Precisa de Python 3.11 ou mais novo.

```bash
python -m venv .venv
source .venv/bin/activate        # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Cada rodada abre numa janela (ESC fecha):

```bash
python -m arena.minimax                  # Rodada 1
python -m arena.pathfinding              # Rodada 2
python -m arena.flappy                   # Rodada 3
python -m arena.pong                     # Rodada 4
python -m arena.snake --cena reforco     # Rodada 5
python -m arena.snake                    # Grande final
python -m arena.snake.benchmark          # Placar: 100 partidas para cada IA
```

Quer ser o professor da Rodada 4? Jogue com W/S (ou as setas) e depois veja a IA imitando você:

```bash
python -m arena.pong --gravar meus_dados.npz
python -m arena.pong --imitar meus_dados.npz
```

Para gravar os vídeos em `.mp4` é preciso ter o [ffmpeg](https://ffmpeg.org) instalado:

```bash
python -m arena.minimax --video videos/1_minimax.mp4   # um vídeo
python -m arena.record_all                              # os seis vídeos da palestra
```

## O que cada IA faz

**Minimax** (`arena/minimax/search.py`). Imagina todas as jogadas possíveis, supondo que o
adversário sempre responde da melhor forma para ele. A poda alfa-beta deixa de examinar ramos
que não podem mudar a decisão: no Jogo da Velha, a árvore completa tem 549.945 posições; com
poda, a IA decide examinando 20.865. O Ligue 4 é grande demais para ir até o fim, então a busca
para em 6 lances e usa uma função de avaliação (`ConnectFour.heuristic`).

**A\*** (`arena/pathfinding/search.py`). Examina primeiro a casa com menor `f = g + h`: `g` é o
quanto já andou e `h` é um palpite de quanto falta (distância em "quarteirões"). Como o palpite
nunca exagera, o caminho encontrado é o mais curto, igual ao da busca em largura, mas examinando
bem menos casas.

**Neuroevolução** (`arena/shared/genetic.py`, `arena/flappy`). Cada pássaro tem uma rede neural
3-4-1 (21 números). Ninguém treina a rede: a geração seguinte nasce dos pássaros que sobreviveram
mais tempo, com cruzamento e pequenas mutações.

**Imitação** (`arena/pong/imitation.py`). Um professor joga e cada quadro vira um exemplo
(o que se vê na tela → tecla escolhida). Uma rede neural do scikit-learn aprende a prever a tecla.
Ela joga como o professor, inclusive com os vícios dele.

**Q-Learning** (`arena/snake/qlearning.py`). A cobra enxerga 11 perguntas de sim ou não
(`arena/snake/features.py`), ou seja, 2.048 situações. Uma tabela guarda o valor de cada ação
em cada situação e é corrigida a cada passo:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max Q(s', a') − Q(s, a) ]
```

## Placar do grande final

Gerado por `python -m arena.snake.benchmark`: as mesmas 100 partidas para todas as IAs,
tabuleiro 10 × 10 (máximo de 97 maçãs).

| IA | Preparo | Maçãs (média) | Melhor partida | Zerou o tabuleiro |
|---|---|---|---|---|
| Regras if/else | algumas linhas de regra | 19,3 | 35 | 0 de 100 |
| A* até a maçã | nenhum treino | 30,4 | 51 | 0 de 100 |
| Ciclo hamiltoniano | nenhum treino | 97,0 | 97 | 100 de 100 |
| Neuroevolução | 80 gerações | 26,0 | 54 | 0 de 100 |
| Imitação (aluno do A*) | 60 partidas do A* | 19,6 | 37 | 0 de 100 |
| Q-Learning | 3.000 partidas | 17,2 | 33 | 0 de 100 |

A lição: um algoritmo clássico bem escolhido pode vencer todas as IAs que aprendem.
Aprender compensa quando ninguém sabe escrever a regra.

## Arquitetura

O código segue uma arquitetura hexagonal simples:

- **Domínio** — regras dos jogos e algoritmos de IA (`game.py`, `search.py`, `qlearning.py`…).
  Python puro + NumPy (+ scikit-learn na imitação). Não importa `pygame`.
- **Adaptadores de saída** — `arena/view/output.py`: a mesma cena vai para uma janela
  (`WindowSink`) ou para um `.mp4` (`VideoSink`). As cenas não sabem para onde os quadros vão.
- **Adaptador de entrada** — `arena/pong/keyboard.py`: a raquete controlada pelo teclado.
- **Cenas** — `scene.py` de cada rodada: só desenham o que o domínio calcula.

`tests/test_architecture.py` garante que nenhum módulo de domínio importa `pygame`.

```
arena/
├── shared/        rede neural, algoritmo genético e formatação
├── view/          tema, desenho, janela/vídeo e linha de comando
├── minimax/       Rodada 1
├── pathfinding/   Rodada 2
├── flappy/        Rodada 3
├── pong/          Rodada 4
└── snake/         Rodada 5 e grande final
modelos/           IAs do Snake já treinadas (apague para treinar de novo)
tests/             testes automatizados (pytest)
videos/            vídeos usados na palestra
```

## Testes e qualidade

```bash
pip install pytest ruff mypy
python -m pytest        # 39 testes, ~10 s
ruff check . && mypy arena
```

Alguns exemplos do que os testes conferem: a árvore completa do Jogo da Velha tem 549.945
posições; a poda nunca muda a jogada escolhida; o A* sempre acha um caminho tão curto quanto o da
busca em largura; o ciclo hamiltoniano passa por todas as casas e sempre zera o jogo; o Q-Learning
melhora com o treino.

## Inspiração

O formato "uma IA aprendendo a jogar" vem do canal
[Universo Programado](https://www.youtube.com/@UniversoProgramado), de Victor Dias, que me fez
escolher computação.

## Licenças

Fontes JetBrains Mono e Space Grotesk sob a SIL Open Font License (arquivos em `arena/view/fonts`).
