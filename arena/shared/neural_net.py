"""Rede neural minúscula (só a passagem para frente), usada pela neuroevolução.

Na neuroevolução ninguém calcula gradiente: os pesos de cada rede são um "genoma"
(um vetor de números) que o algoritmo genético cruza e muta. Por isso a rede só
precisa saber transformar entradas em saídas.
"""

from collections.abc import Iterator

import numpy as np

LayerSizes = tuple[int, ...]


def genome_length(layer_sizes: LayerSizes) -> int:
    """Quantos números (pesos + vieses) uma rede com essas camadas tem."""
    return sum((n_in + 1) * n_out for n_in, n_out in zip(layer_sizes, layer_sizes[1:], strict=False))


def forward(genomes: np.ndarray, inputs: np.ndarray, layer_sizes: LayerSizes) -> np.ndarray:
    """Roda várias redes de uma vez: ``genomes`` (P, G) e ``inputs`` (P, entradas) → (P, saídas)."""
    return _activations(genomes, inputs, layer_sizes)[-1]


def layer_activations(genome: np.ndarray, inputs: np.ndarray, layer_sizes: LayerSizes) -> list[np.ndarray]:
    """Valores de cada camada de UMA rede (entradas, ocultas e saída) — usado para desenhá-la."""
    return [layer[0] for layer in _activations(genome[np.newaxis], inputs[np.newaxis], layer_sizes)]


def _activations(genomes: np.ndarray, inputs: np.ndarray, layer_sizes: LayerSizes) -> list[np.ndarray]:
    """Camadas ocultas usam tanh; a camada de saída é linear."""
    layers = [inputs]
    last_layer = len(layer_sizes) - 2
    for index, (weights, biases) in enumerate(_unpack(genomes, layer_sizes)):
        values = np.einsum("pi,pio->po", layers[-1], weights) + biases
        layers.append(np.tanh(values) if index < last_layer else values)
    return layers


def _unpack(genomes: np.ndarray, layer_sizes: LayerSizes) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Fatia cada genoma em (pesos, vieses) de cada camada, na ordem."""
    offset = 0
    for n_in, n_out in zip(layer_sizes, layer_sizes[1:], strict=False):
        weights = genomes[:, offset : offset + n_in * n_out].reshape(-1, n_in, n_out)
        offset += n_in * n_out
        biases = genomes[:, offset : offset + n_out]
        offset += n_out
        yield weights, biases
