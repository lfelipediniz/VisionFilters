# analise visual em frequencia pra Parte I
#
# aqui eu uso np.fft apenas pra visualizar e interpretar os filtros convolucionais
# da Parte I. a implementacao dos filtros em si continua em convolution.py, com
# convolucao manual no dominio espacial.
#
# na Parte II a DFT e a IDFT sao implementadas manualmente em fourier_manual.py,
# sem depender de np.fft

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.image_utils import normalize_0_255, to_uint8


def _as_2d_float(values, name):
    # garante que a entrada seja uma matriz 2D numerica
    values = np.asarray(values, dtype=float)

    if values.ndim != 2:
        raise ValueError(f"{name} deve ser uma matriz 2D.")

    return values


def fft2_magnitude(img):
    """Calcula |F(u,v)| de uma imagem usando np.fft.fft2.

    Uso restrito a analise visual da Parte I. Essa funcao nao aplica filtro na
    imagem, apenas revela a distribuicao de frequencias pra gente visualizar
    o conteudo espectral.
    """
    img = _as_2d_float(img, "img")
    spectrum = np.fft.fft2(img)
    return np.abs(spectrum)


def centered_fft_magnitude(img):
    """Calcula a magnitude com baixas frequencias centralizadas por fftshift.

    Em aula vimos que o fftshift move a componente DC pro centro da imagem,
    facilitando a interpretacao visual do espectro.
    """
    img = _as_2d_float(img, "img")
    spectrum = np.fft.fft2(img)
    centered_spectrum = np.fft.fftshift(spectrum)
    return np.abs(centered_spectrum)


def log_magnitude(magnitude):
    # aplica log(1 + magnitude) pra comprimir a escala visual do espectro
    magnitude = np.asarray(magnitude, dtype=float)
    return np.log1p(magnitude)


def normalize_spectrum(spectrum):
    # normaliza um espectro pra visualizacao em escala [0, 255]
    spectrum = np.asarray(spectrum, dtype=float)
    finite_mask = np.isfinite(spectrum)

    if not finite_mask.any():
        return np.zeros_like(spectrum, dtype=np.uint8)

    spectrum_min = spectrum[finite_mask].min()
    spectrum_max = spectrum[finite_mask].max()

    if np.isclose(spectrum_max, spectrum_min, rtol=1e-12, atol=1e-12):
        if spectrum_max > 0:
            return np.full_like(spectrum, 255, dtype=np.uint8)

        return np.zeros_like(spectrum, dtype=np.uint8)

    return to_uint8(normalize_0_255(spectrum))


def magnitude_spectrum(img):
    # gera o espectro visual completo (fft2 + fftshift + log + normalizacao)
    magnitude = centered_fft_magnitude(img)
    log_spectrum = log_magnitude(magnitude)
    return normalize_spectrum(log_spectrum)


def center_kernel_in_shape(kernel, shape):
    """Centraliza um kernel pequeno numa matriz do tamanho da imagem.

    A centralizacao facilita a inspecao visual. Pra resposta em frequencia em
    magnitude, deslocar o kernel muda a fase mas nao muda |H(u,v)|, entao
    essa escolha eh adequada pra visualizacao.
    """
    kernel = _as_2d_float(kernel, "kernel")

    if len(shape) != 2:
        raise ValueError("shape deve ter duas dimensoes (altura, largura).")

    img_h, img_w = int(shape[0]), int(shape[1])
    kernel_h, kernel_w = kernel.shape

    if img_h <= 0 or img_w <= 0:
        raise ValueError("shape deve ter altura e largura positivas.")

    if kernel_h > img_h or kernel_w > img_w:
        raise ValueError("O kernel deve caber dentro da matriz da imagem.")

    kernel_pad = np.zeros((img_h, img_w), dtype=float)
    start_i = (img_h - kernel_h) // 2
    start_j = (img_w - kernel_w) // 2
    kernel_pad[start_i:start_i + kernel_h, start_j:start_j + kernel_w] = kernel
    return kernel_pad


def kernel_frequency_response(kernel, shape):
    """Calcula a resposta em frequencia visual do kernel |H(u,v)|.

    Aqui np.fft serve so pra responder a pergunta do relatorio sobre quais
    frequencias o kernel tende a suprimir ou realcar. O filtro em si continua
    sendo aplicado por convolve2d no dominio espacial.
    """
    kernel_pad = center_kernel_in_shape(kernel, shape)
    response = np.fft.fft2(kernel_pad)
    centered_response = np.fft.fftshift(response)
    magnitude = np.abs(centered_response)
    return normalize_spectrum(log_magnitude(magnitude))


def combined_kernel_frequency_response(kernels, shape):
    """Combina respostas de varios kernels por energia de magnitude.

    Essa funcao eh util pro Sobel magnitude. A imagem final do Sobel nao vem
    de um unico kernel linear porque combina duas respostas por sqrt(Gi^2 + Gj^2).
    Mesmo assim a combinacao sqrt(|Hi|^2 + |Hj|^2) mostra quais frequencias e
    direcoes o processo tende a realcar.
    """
    if len(kernels) == 0:
        raise ValueError("Informe pelo menos um kernel.")

    combined_power = None

    for kernel in kernels:
        kernel_pad = center_kernel_in_shape(kernel, shape)
        response = np.fft.fft2(kernel_pad)
        centered_response = np.fft.fftshift(response)
        power = np.abs(centered_response) ** 2

        if combined_power is None:
            combined_power = power
        else:
            combined_power = combined_power + power

    magnitude = np.sqrt(combined_power)
    return normalize_spectrum(log_magnitude(magnitude))


def build_frequency_analysis_images(original_img, kernel, filtered_img):
    # monta as cinco imagens usadas na figura de analise da Parte I
    original_img = _as_2d_float(original_img, "original_img")
    filtered_img = _as_2d_float(filtered_img, "filtered_img")

    if original_img.shape != filtered_img.shape:
        raise ValueError("original_img e filtered_img devem ter o mesmo tamanho.")

    return [
        to_uint8(original_img),
        magnitude_spectrum(original_img),
        kernel_frequency_response(kernel, original_img.shape),
        to_uint8(filtered_img),
        magnitude_spectrum(filtered_img),
    ]


def build_frequency_analysis_images_from_response(original_img, response_img, filtered_img):
    # monta a figura quando a resposta visual ja foi calculada separadamente
    original_img = _as_2d_float(original_img, "original_img")
    response_img = _as_2d_float(response_img, "response_img")
    filtered_img = _as_2d_float(filtered_img, "filtered_img")

    if original_img.shape != filtered_img.shape:
        raise ValueError("original_img e filtered_img devem ter o mesmo tamanho.")

    if original_img.shape != response_img.shape:
        raise ValueError("response_img deve ter o mesmo tamanho da imagem original.")

    return [
        to_uint8(original_img),
        magnitude_spectrum(original_img),
        to_uint8(response_img),
        to_uint8(filtered_img),
        magnitude_spectrum(filtered_img),
    ]


def save_frequency_analysis_figure(
    output_path,
    original_img,
    kernel,
    filtered_img,
    filter_name,
    params=None,
    figsize=(15, 4),
    dpi=220,
):
    # salva a figura com imagem, espectro, |H|, resultado e espectro resultante
    images = build_frequency_analysis_images(original_img, kernel, filtered_img)
    params = "" if params is None else f"\n{params}"

    titles = [
        "imagem original",
        "espectro da original",
        "resposta |H(u,v)|",
        "imagem filtrada",
        "espectro da filtrada",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=figsize)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"{filter_name}{params}")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    return output_path


def save_frequency_analysis_figure_from_response(
    output_path,
    original_img,
    response_img,
    filtered_img,
    filter_name,
    params=None,
    response_title="resposta em frequencia",
    figsize=(15, 4),
    dpi=220,
):
    # salva a analise quando a coluna de resposta nao vem de um kernel unico
    images = build_frequency_analysis_images_from_response(
        original_img,
        response_img,
        filtered_img,
    )
    params = "" if params is None else f"\n{params}"

    titles = [
        "imagem original",
        "espectro da original",
        response_title,
        "imagem filtrada",
        "espectro da filtrada",
    ]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 5, figsize=figsize)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(f"{filter_name}{params}")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)

    return output_path


def pad_kernel_to_shape(kernel, shape):
    # alias pra manter a nomenclatura usada no planejamento
    return center_kernel_in_shape(kernel, shape)
