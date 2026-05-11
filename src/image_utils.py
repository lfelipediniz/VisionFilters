# funcoes auxiliares pra leitura, conversao e salvamento de imagens
# antes de aplicar os filtros convolucionais precisa padronizar a entrada.
# centralizei aqui operacoes simples como leitura com imageio, conversao pra
# tons de cinza, normalizacao min-max, recorte central e salvamento

from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np


def load_image(path):
    # le uma imagem do disco usando imageio.v3
    return iio.imread(path)


def rgb_to_gray(img):
    """Converte RGB/RGBA pra tons de cinza pela formula de luminosidade.

    Usa pesos perceptuais onde o canal verde influencia mais a percepcao de
    brilho, seguido do vermelho e depois do azul. Essa formula foi a mesma
    que a gente viu em aula pra converter pra escala de cinza.
    """
    img = np.asarray(img)

    if img.ndim == 2:
        return img.astype(float)

    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError("A imagem deve ser 2D, RGB ou RGBA.")

    rgb = img[..., :3].astype(float)
    return 0.21 * rgb[..., 0] + 0.72 * rgb[..., 1] + 0.07 * rgb[..., 2]


def luminosity(img):
    # mesma conversao pra cinza, mas ja devolve em uint8
    return to_uint8(rgb_to_gray(img))


def norm_minmax(img, C=255, m=0):
    """Normaliza uma imagem por min-max.

    Mapeia o menor valor pra 0 e o maior pra C, espalhando os valores no
    intervalo. Essa normalizacao eh a mesma que a gente usou em aula pra
    garantir que a imagem ocupe toda a faixa dinamica disponivel.
    """
    img = np.asarray(img, dtype=float)
    finite_mask = np.isfinite(img)

    if not finite_mask.any():
        return np.zeros_like(img)

    img_min = img[finite_mask].min()
    img_max = img[finite_mask].max()

    if img_max == img_min:
        return np.zeros_like(img)

    # valores nao finitos nao devem dominar a escala da imagem
    safe_img = np.where(finite_mask, img, img_min)
    normalized = (safe_img - img_min) / (img_max - img_min)
    return normalized * C - m


def normalize_0_255(img):
    # normaliza qualquer matriz numerica pro intervalo [0, 255]
    return norm_minmax(img, C=255, m=0)


def to_uint8(img, normalize=False):
    """Converte pra uint8 com seguranca.

    Quando normalize=True, primeiro espalha os valores pra [0, 255]. Quando
    normalize=False, apenas troca NaN/inf, recorta a faixa e converte.
    """
    if normalize:
        img = normalize_0_255(img)

    img = np.asarray(img, dtype=float)
    img = np.nan_to_num(img, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def clip_uint8(img):
    # recorta valores pra [0, 255] e converte pra uint8
    return to_uint8(img, normalize=False)


def read_gray(path):
    # le uma imagem e devolve em tons de cinza
    return luminosity(load_image(path))


def center_crop(img, crop_shape):
    """Recorta o centro da imagem.

    crop_shape pode ser um inteiro pra recorte quadrado ou uma tupla
    (altura, largura). Se o recorte pedido for maior que a imagem, usa o
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
    # salva uma imagem individual, criando a pasta de destino se precisar
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, to_uint8(img, normalize=normalize))


def show_bw(img, title=None):
    # mostra uma imagem em tons de cinza com matplotlib
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
    dpi=220,
    interpolation="nearest",
):
    # salva uma figura comparativa com varias imagens lado a lado
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
            ax.imshow(img, cmap=cmap, vmin=0, vmax=255, interpolation=interpolation)
        else:
            ax.imshow(to_uint8(img), interpolation=interpolation)

        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    for ax in axes[n_images:]:
        ax.axis("off")

    if main_title is not None:
        fig.suptitle(main_title)
        # tight_layout nao sabe do suptitle, entao reservamos espaco no topo
        # proporcional ao numero de linhas do titulo pra nao sobrepor as imagens
        n_lines = len(main_title.split("\n"))
        top = 1.0 - 0.08 * n_lines
        fig.tight_layout(pad=0.8, rect=[0, 0, 1, top])
    else:
        fig.tight_layout(pad=0.8)

    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def ensure_output_dirs(base_dir="outputs"):
    # garante que as pastas de saida dos experimentos existam
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
    # lista imagens disponiveis pra os experimentos
    imgs_dir = Path(imgs_dir)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]

    paths = []
    for extension in extensions:
        paths.extend(imgs_dir.glob(extension))

    return sorted(paths)
