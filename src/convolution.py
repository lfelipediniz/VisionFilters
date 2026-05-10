"""Base para convolucao manual.

Conceito:
A convolucao posiciona um kernel sobre cada pixel, multiplica os valores da
vizinhanca pelos pesos do kernel e soma tudo para gerar o novo pixel. Esta
implementacao sera manual, sem chamadas prontas como cv2.filter2D,
scipy.signal.convolve2d ou ndimage.convolve.
"""

import numpy as np


def validate_kernel(kernel):
    """Confere se o kernel e bidimensional e tem dimensoes impares."""
    kernel = np.asarray(kernel, dtype=float)

    if kernel.ndim != 2:
        raise ValueError("O kernel deve ser uma matriz 2D.")

    k_h, k_w = kernel.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("O kernel deve ter altura e largura impares.")

    return kernel


def pad_image(img, pad_h, pad_w, mode="constant"):
    """Prepara a imagem para tratar bordas durante a convolucao.

    Sera completada quando compararmos os efeitos dos paddings:
    - constant: completa com zero
    - edge: repete a borda
    - reflect: espelha a imagem
    - wrap: conecta lados opostos
    """
    raise NotImplementedError("Implementar os modos de padding na etapa de convolucao.")


def conv_op(i, j, img, kernel):
    """Calcula um unico pixel filtrado a partir da vizinhanca de (i, j)."""
    raise NotImplementedError("Implementar a operacao local da convolucao manual.")


def convolve_manual(img, kernel, padding="constant"):
    """Aplica convolucao manual em uma imagem 2D.

    Esta sera a funcao central da Parte I. Os filtros obrigatorios deverao
    chamar esta funcao em vez de usar convolucoes prontas de bibliotecas.
    """
    raise NotImplementedError("Implementar a convolucao manual completa.")

