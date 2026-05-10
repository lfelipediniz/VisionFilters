"""Geradores de kernels para filtros convolucionais.

Conceito:
Um kernel e uma pequena matriz de pesos. Cada filtro muda esses pesos para
produzir deslocamento, suavizacao, deteccao de bordas ou realce de detalhes.
"""

import numpy as np


def shift_kernel(size, direction="down_right"):
    """Kernel de deslocamento.

    Ideia: colocar um unico peso 1 fora do centro desloca a imagem porque cada
    pixel passa a copiar um vizinho.
    """
    raise NotImplementedError("Implementar o kernel de shift nos proximos passos.")


def box_kernel(size):
    """Kernel caixa/media.

    Ideia: todos os vizinhos recebem o mesmo peso, entao o resultado e a media
    local e tende a suavizar ruido e detalhes finos.
    """
    raise NotImplementedError("Implementar o kernel de media nos proximos passos.")


def gaussian_kernel(size, sigma):
    """Kernel gaussiano.

    Ideia: vizinhos proximos do centro pesam mais que os distantes, suavizando
    de modo mais natural que o filtro de caixa.
    """
    raise NotImplementedError("Implementar o kernel gaussiano nos proximos passos.")


def laplace_kernel(kind=8):
    """Kernel de Laplace.

    Ideia: aproxima a segunda derivada da imagem, destacando mudancas bruscas
    de intensidade, especialmente bordas.
    """
    raise NotImplementedError("Implementar o kernel de Laplace nos proximos passos.")


def sobel_kernels():
    """Kernels de Sobel nas direcoes horizontal e vertical.

    Ideia: combina uma derivada em uma direcao com suavizacao na outra para
    medir bordas horizontais e verticais.
    """
    raise NotImplementedError("Implementar os kernels de Sobel nos proximos passos.")


def emboss_kernel(direction="down_right"):
    """Kernel criativo de relevo/emboss.

    Ideia: pesos negativos de um lado e positivos do outro simulam uma luz
    direcional. Regioes uniformes ficam quase neutras e bordas viram relevo.
    """
    raise NotImplementedError("Implementar o kernel de relevo no filtro criativo.")


def identity_kernel(size=3):
    """Kernel identidade, util para testes da arquitetura."""
    kernel = np.zeros((size, size), dtype=float)
    kernel[size // 2, size // 2] = 1
    return kernel
