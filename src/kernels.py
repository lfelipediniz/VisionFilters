# geradores de kernels para filtros convolucionais
#
# um kernel eh basicamente uma pequena matriz de pesos. cada filtro muda esses
# pesos pra produzir efeitos diferentes como deslocamento, suavizacao, deteccao
# de bordas ou realce de detalhes. em aula a gente viu varios desses kernels
# e como eles se comportam no dominio espacial e em frequencia

import numpy as np


def _validate_odd_size(size):
    # confere se o tamanho do kernel eh impar e positivo
    size = int(size)

    if size <= 0:
        raise ValueError("O tamanho do kernel deve ser positivo.")

    if size % 2 == 0:
        raise ValueError("O tamanho do kernel deve ser impar.")

    return size


def shift_kernel(size, di=1, dj=1):
    """Kernel de deslocamento.

    A ideia eh colocar um unico peso 1 fora do centro, assim cada pixel passa
    a copiar um vizinho e a imagem inteira se desloca. Em aula vimos que isso
    muda a fase no dominio da frequencia mas nao altera a magnitude.
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
    """Kernel caixa (filtro de media).

    Todos os vizinhos recebem o mesmo peso, entao o resultado eh a media local.
    Isso tende a suavizar ruido e detalhes finos. Em frequencia ele se comporta
    como um passa-baixa com formato de sinc, que a gente viu em aula.
    """
    size = _validate_odd_size(size)
    return np.ones((size, size), dtype=float) / float(size**2)


def gaussian_kernel(size, sigma):
    """Kernel gaussiano.

    Vizinhos proximos do centro pesam mais que os distantes, suavizando de modo
    mais natural que o filtro de caixa. A gaussiana eh a unica funcao que eh
    igual no dominio espacial e no dominio da frequencia, entao nao gera aqueles
    lobos laterais que aparecem no sinc do filtro de media.
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

    Aproxima a segunda derivada da imagem, destacando mudancas bruscas de
    intensidade (bordas). Em aula vimos que ele funciona como um filtro
    passa-alta, zerando regioes uniformes e respondendo forte onde tem transicao.
    """
    if kind != 8:
        raise ValueError("Nesta Parte I, use o Laplace 3x3 com 8 vizinhos.")

    return np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1],
    ], dtype=float)


def sobel_i_kernel():
    # kernel de Sobel pra variacao vertical, realca bordas horizontais
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ], dtype=float)


def sobel_j_kernel():
    # kernel de Sobel pra variacao horizontal, realca bordas verticais
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ], dtype=float)


def sobel_kernels():
    """Retorna os dois kernels de Sobel (horizontal e vertical).

    A ideia do Sobel eh combinar uma derivada em uma direcao com suavizacao
    na outra, pra medir bordas de forma mais robusta que uma derivada simples.
    Em aula vimos que o gradiente combina as duas componentes pela magnitude.
    """
    return sobel_i_kernel(), sobel_j_kernel()


def emboss_kernel():
    """Kernel criativo de relevo (emboss).

    Pesos negativos de um lado e positivos do outro simulam uma luz direcional.
    Regioes uniformes ficam quase neutras e bordas viram relevo. Escolhi esse
    como meu filtro criativo porque o efeito visual eh bem interessante.
    """
    return np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2],
    ], dtype=float)


def identity_kernel(size=3):
    # kernel identidade, util pra testes da arquitetura
    size = _validate_odd_size(size)
    kernel = np.zeros((size, size), dtype=float)
    kernel[size // 2, size // 2] = 1
    return kernel
