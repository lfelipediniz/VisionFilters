"""Geradores de kernels para filtros convolucionais.

Conceito:
Um kernel e uma pequena matriz de pesos. Cada filtro muda esses pesos para
produzir deslocamento, suavizacao, deteccao de bordas ou realce de detalhes.
"""

import numpy as np


def _validate_odd_size(size):
    """Confere se o tamanho do kernel e impar e positivo."""
    size = int(size)

    if size <= 0:
        raise ValueError("O tamanho do kernel deve ser positivo.")

    if size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser impar.")

    return size


def shift_kernel(size, di=1, dj=1):
    """Kernel de deslocamento.

    Ideia: colocar um unico peso 1 fora do centro desloca a imagem porque cada
    pixel passa a copiar um vizinho.
    """
    size = _validate_odd_size(size)
    di = int(di)
    dj = int(dj)

    center = size // 2
    pos_i = center + di
    pos_j = center + dj

    if pos_i < 0 or pos_i >= size or pos_j < 0 or pos_j >= size:
        raise ValueError("O deslocamento deve caber dentro do kernel.")

    kernel = np.zeros((size, size), dtype=float)
    kernel[pos_i, pos_j] = 1.0
    return kernel


def box_kernel(size):
    """Kernel caixa/media.

    Ideia: todos os vizinhos recebem o mesmo peso, entao o resultado e a media
    local e tende a suavizar ruido e detalhes finos.
    """
    size = _validate_odd_size(size)
    return np.ones((size, size), dtype=float) / float(size**2)


def gaussian_kernel(size, sigma):
    """Kernel gaussiano.

    Ideia: vizinhos proximos do centro pesam mais que os distantes, suavizando
    de modo mais natural que o filtro de caixa.
    """
    size = _validate_odd_size(size)
    sigma = float(sigma)

    if sigma <= 0:
        raise ValueError("Sigma deve ser positivo.")

    center = size // 2
    coords = np.arange(size) - center
    kernel = np.zeros((size, size), dtype=float)

    for i, x in enumerate(coords):
        for j, y in enumerate(coords):
            kernel[i, j] = np.exp(-((x**2 + y**2) / (2 * sigma**2)))

    return kernel / kernel.sum()


def laplace_kernel(kind=8):
    """Kernel de Laplace.

    Ideia: aproxima a segunda derivada da imagem, destacando mudancas bruscas
    de intensidade, especialmente bordas.
    """
    if kind != 8:
        raise ValueError("Nesta Parte I, use o Laplace 3x3 com 8 vizinhos.")

    return np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ], dtype=float)


def sobel_i_kernel():
    """Kernel de Sobel para variacao vertical.

    Realca mudancas ao longo do eixo i, isto e, bordas horizontais.
    """
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=float)


def sobel_j_kernel():
    """Kernel de Sobel para variacao horizontal.

    Realca mudancas ao longo do eixo j, isto e, bordas verticais.
    """
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)


def sobel_kernels():
    """Kernels de Sobel nas direcoes horizontal e vertical.

    Ideia: combina uma derivada em uma direcao com suavizacao na outra para
    medir bordas horizontais e verticais.
    """
    return sobel_i_kernel(), sobel_j_kernel()


def emboss_kernel():
    """Kernel criativo de relevo/emboss.

    Ideia: pesos negativos de um lado e positivos do outro simulam uma luz
    direcional. Regioes uniformes ficam quase neutras e bordas viram relevo.
    """
    return np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ], dtype=float)


def identity_kernel(size=3):
    """Kernel identidade, util para testes da arquitetura."""
    size = _validate_odd_size(size)
    kernel = np.zeros((size, size), dtype=float)
    kernel[size // 2, size // 2] = 1
    return kernel
