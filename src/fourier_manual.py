"""DFT e IDFT manuais para a Parte II.

Este modulo nao usa transformadas prontas de Fourier. A ideia didatica e
mostrar que a transformada 2D pode ser calculada por separabilidade: uma DFT
nas linhas e outra nas colunas, usando multiplicacao de matrizes.
"""

import numpy as np


def dft_matrix(N):
    """Cria a matriz da DFT com W[k, n] = exp(-j * 2*pi*k*n/N)."""
    N = int(N)

    if N <= 0:
        raise ValueError("N deve ser positivo.")

    k = np.arange(N).reshape((N, 1))
    n = np.arange(N).reshape((1, N))
    return np.exp(-1j * 2 * np.pi * k * n / N)


def idft_matrix(N):
    """Cria a matriz da IDFT usando o conjugado da DFT dividido por N."""
    return np.conjugate(dft_matrix(N)) / int(N)


def dft1_manual(signal):
    """Calcula a DFT 1D por multiplicacao matricial."""
    signal = np.asarray(signal, dtype=complex)

    if signal.ndim != 1:
        raise ValueError("O sinal deve ser 1D.")

    W = dft_matrix(signal.shape[0])
    return W @ signal


def idft1_manual(F):
    """Reconstroi um sinal 1D a partir da DFT manual."""
    F = np.asarray(F, dtype=complex)

    if F.ndim != 1:
        raise ValueError("O espectro deve ser 1D.")

    W_inv = idft_matrix(F.shape[0])
    reconstructed = W_inv @ F
    return reconstructed.real


def dft2_manual(img):
    """Calcula a DFT 2D manual usando separabilidade.

    Para uma imagem MxN:
    F = W_M @ img @ W_N

    Como W[k, n] depende do produto k*n, a matriz e simetrica e serve para a
    multiplicacao pela direita nas colunas.
    """
    img = np.asarray(img, dtype=float)

    if img.ndim != 2:
        raise ValueError("A imagem deve ser 2D.")

    M, N = img.shape
    W_M = dft_matrix(M)
    W_N = dft_matrix(N)
    return W_M @ img @ W_N


def idft2_manual(F):
    """Reconstroi uma imagem 2D usando as matrizes inversas da DFT."""
    F = np.asarray(F, dtype=complex)

    if F.ndim != 2:
        raise ValueError("O espectro deve ser 2D.")

    M, N = F.shape
    W_M_inv = idft_matrix(M)
    W_N_inv = idft_matrix(N)
    reconstructed = W_M_inv @ F @ W_N_inv
    return reconstructed.real


def _shift_axis(values, split, axis):
    """Move a segunda parte de um eixo para antes da primeira."""
    values = np.asarray(values)
    first = np.take(values, indices=np.arange(split, values.shape[axis]), axis=axis)
    second = np.take(values, indices=np.arange(0, split), axis=axis)
    return np.concatenate((first, second), axis=axis)


def fftshift_manual(F):
    """Centraliza as baixas frequencias sem usar funcao pronta de shift."""
    shifted = np.asarray(F)

    for axis, size in enumerate(shifted.shape):
        split = (size + 1) // 2
        shifted = _shift_axis(shifted, split, axis)

    return shifted


def ifftshift_manual(F):
    """Desfaz o deslocamento feito por fftshift_manual."""
    shifted = np.asarray(F)

    for axis, size in enumerate(shifted.shape):
        split = size // 2
        shifted = _shift_axis(shifted, split, axis)

    return shifted


def _normalize_visual(values):
    """Normaliza uma matriz numerica para visualizacao em uint8."""
    values = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(values)

    if not finite_mask.any():
        return np.zeros_like(values, dtype=np.uint8)

    v_min = values[finite_mask].min()
    v_max = values[finite_mask].max()

    if np.isclose(v_max, v_min):
        if v_max > 0:
            return np.full_like(values, 255, dtype=np.uint8)

        return np.zeros_like(values, dtype=np.uint8)

    normalized = (values - v_min) / (v_max - v_min)
    return np.clip(np.rint(normalized * 255), 0, 255).astype(np.uint8)


def magnitude_spectrum(F):
    """Calcula log(1 + abs(F)) centralizado e normalizado para visualizacao."""
    F = np.asarray(F, dtype=complex)
    centered = fftshift_manual(F)
    magnitude = np.log1p(np.abs(centered))
    return _normalize_visual(magnitude)


def center_frequency_mask(shape, radius):
    """Cria uma mascara circular de baixas frequencias no centro do espectro."""
    if len(shape) != 2:
        raise ValueError("shape deve ter duas dimensoes.")

    radius = float(radius)

    if radius < 0:
        raise ValueError("radius deve ser nao negativo.")

    h, w = int(shape[0]), int(shape[1])
    center_i = h // 2
    center_j = w // 2

    i = np.arange(h).reshape((h, 1))
    j = np.arange(w).reshape((1, w))
    distance = np.sqrt((i - center_i) ** 2 + (j - center_j) ** 2)
    return (distance <= radius).astype(float)


def partial_reconstruction(F, radius):
    """Reconstroi a imagem usando apenas baixas frequencias ate certo raio."""
    F = np.asarray(F, dtype=complex)

    if F.ndim != 2:
        raise ValueError("O espectro deve ser 2D.")

    centered = fftshift_manual(F)
    mask = center_frequency_mask(F.shape, radius)
    filtered_centered = centered * mask
    filtered = ifftshift_manual(filtered_centered)
    reconstructed = idft2_manual(filtered)
    return _normalize_visual(reconstructed)


def reconstruction_sequence(F, radii):
    """Retorna varias reconstrucoes parciais para uma lista de raios."""
    return [partial_reconstruction(F, radius) for radius in radii]
