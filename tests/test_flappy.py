import numpy as np

from arena.flappy.evolution import LAYERS, evolve, run_generation
from arena.flappy.world import FlappyConfig
from arena.shared.neural_net import forward, genome_length, layer_activations


def test_bird_network_has_21_numbers():
    assert genome_length(LAYERS) == 21


def test_batch_and_single_network_agree():
    rng = np.random.default_rng(0)
    genomes = rng.normal(size=(5, genome_length(LAYERS)))
    inputs = rng.normal(size=(5, 3))
    batch = forward(genomes, inputs, LAYERS)
    single = np.array([layer_activations(genomes[i], inputs[i], LAYERS)[-1] for i in range(5)])
    assert np.allclose(batch, single)


def test_a_bird_that_never_flaps_hits_the_floor():
    never_flaps = np.full((1, genome_length(LAYERS)), 0.0)
    never_flaps[0, -1] = -10.0  # viés da saída muito negativo: nunca pula
    *_, world = run_generation(never_flaps, FlappyConfig(), pipe_seed=0)
    assert not world.flock.alive.any()
    assert world.flock.pipes_passed[0] == 0


def test_evolution_is_reproducible_and_improves():
    first = evolve(12, seed=1)
    second = evolve(12, seed=1)
    assert [r.best_pipes for r in first] == [r.best_pipes for r in second]
    assert first[-1].best_pipes > first[0].best_pipes
