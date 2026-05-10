"""Filtros e processos espaciais da Parte I.

Conceito:
Os filtros espaciais operam diretamente nos pixels. Aqui ficara a camada que
combina kernels com a convolucao manual e organiza processos como nitidez,
unsharp mask e o filtro criativo.
"""


def apply_shift(img, size=3, direction="down_right", padding="constant"):
    """Aplica o filtro de shift usando convolucao manual."""
    raise NotImplementedError("Conectar shift_kernel com convolve_manual.")


def apply_box_blur(img, size=3, padding="constant"):
    """Aplica o filtro caixa/media usando convolucao manual."""
    raise NotImplementedError("Conectar box_kernel com convolve_manual.")


def apply_gaussian_blur(img, size=5, sigma=1.0, padding="constant"):
    """Aplica o filtro gaussiano usando convolucao manual."""
    raise NotImplementedError("Conectar gaussian_kernel com convolve_manual.")


def apply_laplace(img, kind=8, padding="constant"):
    """Aplica o filtro de Laplace usando convolucao manual."""
    raise NotImplementedError("Conectar laplace_kernel com convolve_manual.")


def apply_sobel(img, padding="constant"):
    """Aplica Sobel e devolve componentes e magnitude do gradiente."""
    raise NotImplementedError("Conectar sobel_kernels com convolve_manual.")


def sharpen_laplace(img, alpha=1.0, padding="constant"):
    """Aumenta nitidez combinando imagem original e resposta do Laplace."""
    raise NotImplementedError("Implementar aumento de nitidez com Laplace.")


def unsharp_mask(img, blur_size=5, sigma=1.0, amount=1.0, padding="constant"):
    """Aumenta nitidez subtraindo uma versao borrada da imagem original."""
    raise NotImplementedError("Implementar mascara de des-nitidez.")


def apply_emboss(img, direction="down_right", padding="constant"):
    """Aplica o filtro criativo de relevo usando convolucao manual."""
    raise NotImplementedError("Conectar emboss_kernel com convolve_manual.")


def creative_convolution_process(img, padding="constant"):
    """Executa o processo criativo escolhido para a Parte I.

    Plano atual: usar relevo/emboss, que e simples de explicar e deixa claro
    como um kernel direcional realca transicoes de intensidade.
    """
    raise NotImplementedError("Definir e implementar o processo criativo.")
