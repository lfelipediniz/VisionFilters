# filtros e processos espaciais da Parte I
#
# os filtros espaciais operam diretamente nos pixels, sem ir pro dominio da
# frequencia. aqui eu combino os kernels de kernels.py com a convolucao manual
# de convolution.py pra gerar os resultados da Parte I

import numpy as np

from src.convolution import convolve2d
from src.image_utils import to_uint8
from src.kernels import (
    box_kernel,
    emboss_kernel,
    gaussian_kernel,
    laplace_kernel,
    shift_kernel,
    sobel_i_kernel,
    sobel_j_kernel,
)


def _as_gray_float(img):
    # garante que os filtros recebam uma imagem 2D numerica
    img = np.asarray(img, dtype=float)

    if img.ndim != 2:
        raise ValueError("Os filtros espaciais esperam uma imagem 2D em escala de cinza.")

    return img


def _apply_kernel(img, kernel, padding):
    # aplica um kernel com a convolucao manual centralizada do projeto
    return convolve2d(_as_gray_float(img), kernel, padding=padding)


def _laplace_response(img, padding):
    # calcula a resposta assinada do Laplace, sem normalizar
    return _apply_kernel(img, laplace_kernel(), padding)


def apply_shift(img, size=3, di=1, dj=1, padding="zero"):
    # desloca a imagem usando um kernel com um unico valor 1
    response = _apply_kernel(img, shift_kernel(size, di, dj), padding)
    return to_uint8(response)


def apply_box_blur(img, size=3, padding="zero"):
    # aplica filtro caixa (media) usando pesos iguais na vizinhanca
    response = _apply_kernel(img, box_kernel(size), padding)
    return to_uint8(response)


def apply_gaussian_blur(img, size=5, sigma=1.0, padding="zero"):
    # aplica filtro gaussiano, que suaviza dando mais peso ao centro
    response = _apply_kernel(img, gaussian_kernel(size, sigma), padding)
    return to_uint8(response)


def apply_laplace(img, padding="zero"):
    """Aplica Laplace e usa valor absoluto pra visualizacao.

    A resposta crua tem valores negativos e positivos. Pro relatorio eu uso o
    valor absoluto porque deixa as bordas claras em fundo escuro, que eh mais
    facil de interpretar do que a resposta assinada em cinza medio.
    """
    response = _laplace_response(img, padding)
    return to_uint8(np.abs(response), normalize=True)


def apply_sobel(img, padding="zero"):
    """Aplica Sobel e combina as direcoes pela magnitude do gradiente.

    Em aula vimos que a magnitude sqrt(Gi^2 + Gj^2) junta as duas componentes
    do gradiente num unico mapa de bordas, independente da direcao.
    """
    grad_i = _apply_kernel(img, sobel_i_kernel(), padding)
    grad_j = _apply_kernel(img, sobel_j_kernel(), padding)

    magnitude = np.sqrt(grad_i**2 + grad_j**2)
    return to_uint8(magnitude, normalize=True)


def sharpen_with_laplace(img, alpha=1.0, padding="zero"):
    """Aumenta nitidez usando imagem_original - alpha * Laplace.

    Como o Laplace eh passa-alta, subtrair ele da imagem original reforca as
    altas frequencias (detalhes e bordas) sem perder as baixas (estrutura geral).
    Esse processo de sharpening com Laplace foi visto em aula.
    """
    img = _as_gray_float(img)
    laplace = _laplace_response(img, padding)
    sharpened = img - alpha * laplace
    return to_uint8(sharpened)


def sharpen_laplace(img, alpha=1.0, padding="zero"):
    # alias mantido pra nomenclatura planejada inicialmente
    return sharpen_with_laplace(img, alpha=alpha, padding=padding)


def unsharp_mask(img, size=5, sigma=1.0, amount=1.0, padding="zero", blur_size=None):
    """Aumenta nitidez somando uma mascara de detalhes a imagem original.

    A ideia eh que borrada = gaussiano(imagem), mascara = imagem - borrada,
    e resultado = imagem + amount * mascara. Basicamente a gente extrai os
    detalhes que o blur removeu e devolve eles pra imagem com um peso controlado.
    Esse processo de unsharp masking eh classico em processamento de imagens.
    """
    if blur_size is not None:
        size = blur_size

    img = _as_gray_float(img)
    blurred = _apply_kernel(img, gaussian_kernel(size, sigma), padding)
    mask = img - blurred
    result = img + amount * mask
    return to_uint8(result)


def apply_emboss(img, padding="zero"):
    # aplica o filtro criativo de relevo (emboss)
    response = _apply_kernel(img, emboss_kernel(), padding)
    return to_uint8(response, normalize=True)


def creative_convolution_process(img, padding="zero"):
    # processo criativo escolhido pra Parte I, que eh o emboss
    return apply_emboss(img, padding=padding)
