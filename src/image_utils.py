"""Funcoes auxiliares para leitura, conversao e salvamento de imagens.

Conceito:
Antes de aplicar filtros convolucionais, padronizamos a entrada. Este arquivo
centraliza operacoes simples: leitura com imageio, conversao para tons de
cinza, normalizacao, recorte central e salvamento de resultados.
"""

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    """Le uma imagem do disco usando imageio.v3."""
    return iio.imread(path)


def rgb_to_gray(img):
    """Converte RGB/RGBA para tons de cinza pela formula de luminosidade.

    A formula usa pesos perceptuais: o canal verde influencia mais a percepcao
    de brilho, seguido do vermelho e depois do azul.
    """
    img = np.asarray(img)

    if img.ndim == 2:
        return img.astype(float)

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("A imagem deve ser 2D, RGB ou RGBA.")

    rgb = img[..., :3].astype(float)
    return 0.21 * rgb[..., 0] + 0.72 * rgb[..., 1] + 0.07 * rgb[..., 2]


def luminosity(img):
    """Versao no estilo da professora: devolve tons de cinza em uint8."""
    return to_uint8(rgb_to_gray(img))


def norm_minmax(img, C=255, m=0):
    """Normaliza uma imagem por min-max, seguindo o padrao usado em aula."""
    img = np.asarray(img, dtype=float)
    finite_mask = np.isfinite(img)

    if not finite_mask.any():
        return np.zeros_like(img)

    img_min = img[finite_mask].min()
    img_max = img[finite_mask].max()

    if img_max == img_min:
        return np.zeros_like(img)

    # Valores nao finitos nao devem dominar a escala da imagem.
    safe_img = np.where(finite_mask, img, img_min)
    normalized = (safe_img - img_min) / (img_max - img_min)
    return normalized * C - m


def normalize_0_255(img):
    """Normaliza qualquer matriz numerica para o intervalo [0, 255]."""
    return norm_minmax(img, C=255, m=0)


def to_uint8(img, normalize=False):
    """Converte para uint8 com seguranca.

    Quando normalize=True, primeiro espalha os valores para [0, 255]. Quando
    normalize=False, apenas troca NaN/inf, recorta a faixa e converte.
    """
    if normalize:
        img = normalize_0_255(img)

    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def clip_uint8(img):
    """Alias didatico: recorta valores para [0, 255] e converte para uint8."""
    return to_uint8(img, normalize=False)


def read_gray(path):
    """Le uma imagem e devolve sua versao em tons de cinza."""
    return luminosity(load_image(path))


def center_crop(img, crop_shape):
    """Recorta o centro da imagem.

    crop_shape pode ser um inteiro, para recorte quadrado, ou uma tupla
    (altura, largura). Se o recorte pedido for maior que a imagem, usamos o
    maior recorte central possivel.
    """
    img = np.asarray(img)

    if isinstance(crop_shape, int):
        crop_h = crop_w = crop_shape
    else:
        crop_h, crop_w = crop_shape

    if crop_h <= 0 or crop_w <= 0:
        raise ValueError("O recorte deve ter altura e largura positivas.")

    h, w = img.shape[:2]
    crop_h = min(int(crop_h), h)
    crop_w = min(int(crop_w), w)

    start_i = (h - crop_h) // 2
    start_j = (w - crop_w) // 2
    return img[start_i:start_i + crop_h, start_j:start_j + crop_w, ...]


def save_image(path, img, normalize=False):
    """Salva uma imagem individual, criando a pasta de destino."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, to_uint8(img, normalize=normalize))


def show_bw(img, title=None):
    """Mostra uma imagem em tons de cinza com matplotlib."""
    plt.imshow(img, cmap="gray", vmin=0, vmax=255)
    plt.axis("off")
    if title is not None:
        plt.title(title)


def save_comparison_figure(
    path,
    images,
    titles=None,
    cols=None,
    figsize=None,
    cmap="gray",
    normalize=False,
    main_title=None,
):
    """Salva uma figura comparativa com varias imagens lado a lado."""
    if len(images) == 0:
        raise ValueError("Informe pelo menos uma imagem para comparar.")

    n_images = len(images)
    titles = titles or [""] * n_images

    if len(titles) != n_images:
        raise ValueError("A quantidade de titulos deve bater com a de imagens.")

    if cols is None:
        cols = n_images

    cols = max(1, int(cols))
    rows = int(np.ceil(n_images / cols))

    if figsize is None:
        figsize = (4 * cols, 4 * rows)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.asarray(axes).reshape(-1)

    for index, (ax, img, title) in enumerate(zip(axes, images, titles)):
        img = normalize_0_255(img) if normalize else img
        img = np.asarray(img)

        if img.ndim == 2:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255)
        else:
            ax.imshow(to_uint8(img))

        ax.set_title(title)
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    if main_title is not None:
        fig.suptitle(main_title)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def ensure_output_dirs(base_dir="outputs"):
    """Garante a existencia das pastas planejadas para os experimentos."""
    folders = [
        "parte1_padding",
        "parte1_filtros",
        "parte1_frequencias",
        "parte2_fourier",
    ]

    base_dir = Path(base_dir)
    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)


def list_images(imgs_dir="imgs"):
    """Lista imagens disponiveis para os proximos experimentos."""
    imgs_dir = Path(imgs_dir)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    paths = []
    for extension in extensions:
        paths.extend(imgs_dir.glob(extension))

    return sorted(paths)
