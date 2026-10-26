import statistics

from arena.snake.classic_agents import HamiltonianAgent, hamiltonian_cycle
from arena.snake.features import STATE_COUNT, state_index
from arena.snake.game import Direction, SnakeGame, Turn
from arena.snake.qlearning import QLearningAgent, train


def test_hamiltonian_cycle_visits_every_cell_once_with_adjacent_steps():
    cycle = hamiltonian_cycle(10)
    assert len(cycle) == 100 and set(cycle.values()) == set(cycle)
    for cell, nxt in cycle.items():
        assert abs(cell[0] - nxt[0]) + abs(cell[1] - nxt[1]) == 1
    cell, seen = (0, 0), set()
    for _ in range(100):
        seen.add(cell)
        cell = cycle[cell]
    assert len(seen) == 100 and cell == (0, 0)


def test_hamiltonian_agent_always_fills_the_board():
    game = SnakeGame(seed=3)
    agent = HamiltonianAgent(10)
    while not game.is_over:
        game.step(agent.choose(game))
    assert game.won and game.score == 97


def test_hitting_the_wall_kills():
    game = SnakeGame(seed=0)
    while game.alive:
        game.step(Direction.UP)
    assert game.cause_of_death == "bateu"


def test_moving_into_the_tail_cell_is_allowed():
    game = SnakeGame(seed=0)
    game.body.clear()
    game.body.extend([(5, 5), (5, 6), (4, 6), (4, 5)])  # cabeça ao lado da ponta do rabo
    game.apple = (0, 0)
    game.direction = Direction.UP
    assert not game.is_deadly((4, 5))


def test_starvation():
    game = SnakeGame(seed=0, starvation_limit=10)
    game.apple = (9, 9)  # longe do quadradinho onde a cobra vai ficar dando voltas
    loop = [Direction.DOWN, Direction.LEFT, Direction.UP, Direction.RIGHT] * 10
    for direction in loop:
        if game.is_over:
            break
        game.step(direction)
    assert game.cause_of_death == "fome"


def test_turns_are_relative_to_heading():
    assert Turn.RIGHT.apply_to(Direction.UP) == Direction.RIGHT
    assert Turn.LEFT.apply_to(Direction.UP) == Direction.LEFT
    assert Turn.STRAIGHT.apply_to(Direction.DOWN) == Direction.DOWN


def test_state_index_fits_the_table():
    assert 0 <= state_index(SnakeGame(seed=0)) < STATE_COUNT


def test_q_learning_gets_better_with_training():
    result = train(300, snapshot_at=(0, 300))

    def mean_score(table):
        agent = QLearningAgent(table, seed=0)
        scores = []
        for seed in range(30):
            game = SnakeGame(seed=50_000 + seed)
            while not game.is_over:
                game.step(agent.choose(game))
            scores.append(game.score)
        return statistics.mean(scores)

    assert mean_score(result.snapshots[300]) > mean_score(result.snapshots[0]) + 2
