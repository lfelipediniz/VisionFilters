"""Ferramentas para observar filtros no dominio das frequencias.

Conceito:
No dominio espacial vemos a imagem pixel a pixel. No dominio das frequencias
vemos quais variacoes sao lentas ou rapidas. Filtros de suavizacao tendem a
suprimir altas frequencias; filtros de borda e nitidez tendem a realca-las.
"""

import numpy as np


def magnitude_spectrum(values):
    """Calcula um espectro de magnitude em escala logaritmica para visualizacao."""
    spectrum = np.fft.fftshift(np.fft.fft2(values))
    return np.log1p(np.abs(spectrum))


def pad_kernel_to_shape(kernel, shape):
    """Centraliza um kernel pequeno em uma matriz do tamanho da imagem.

    Essa funcao segue a ideia do pad_filter usado pela professora e sera base
    para visualizar a resposta dos filtros no dominio das frequencias.
    """
    raise NotImplementedError("Implementar o preenchimento do kernel para frequencias.")


def kernel_frequency_response(kernel, shape):
    """Prepara a resposta em frequencia de um kernel no tamanho da imagem.

    Sera usada para comparar, no relatorio, quais frequencias cada filtro
    suprime ou realca.
    """
    raise NotImplementedError("Implementar resposta em frequencia dos kernels.")
