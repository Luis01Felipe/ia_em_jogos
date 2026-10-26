import random

import pytest

from arena.pathfinding.grid import GridMap, maze_with_loops, open_field
from arena.pathfinding.search import a_star_search, breadth_first_search, find_path


def count_examined(search):
    examined = 0
    while True:
        try:
            next(search)
            examined += 1
        except StopIteration as finished:
            return examined, finished.value


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("build", [maze_with_loops, open_field])
def test_a_star_finds_a_shortest_path_examining_no_more_cells_than_bfs(seed, build):
    grid = build(31, 21, random.Random(seed))
    bfs_examined, bfs_path = count_examined(breadth_first_search(grid.start, grid.goal, grid.neighbors))
    a_star_examined, a_star_path = count_examined(a_star_search(grid.start, grid.goal, grid.neighbors))
    assert len(a_star_path) == len(bfs_path)
    assert a_star_examined <= bfs_examined
    if a_star_path:
        assert a_star_path[0] == grid.start and a_star_path[-1] == grid.goal
        for a, b in zip(a_star_path, a_star_path[1:], strict=False):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            assert grid.is_open(b)


def test_no_path_returns_empty_list():
    walls = frozenset((2, y) for y in range(5))
    grid = GridMap(5, 5, walls, start=(0, 0), goal=(4, 4))
    assert find_path(a_star_search(grid.start, grid.goal, grid.neighbors)) == []
    assert find_path(breadth_first_search(grid.start, grid.goal, grid.neighbors)) == []
