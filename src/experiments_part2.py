# experimentos da Parte II sobre DFT e IDFT manuais
#
# aqui uso somente as funcoes de fourier_manual.py pra transformar e reconstruir
# imagens. as imagens sao reduzidas pra tamanhos menores porque a DFT manual
# por matriz tem custo O(N^2) e fica pesada quando a imagem cresce

from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from src.fourier_manual import (
    dft2_manual,
    idft2_manual,
    magnitude_spectrum,
    partial_reconstruction,
    reconstruction_sequence,
)
from src.image_utils import center_crop, normalize_0_255, read_gray, save_image


DEFAULT_SIZE = 576
DEFAULT_RADII = [18, 36, 72, 108, 180, 288]
CHOSEN_RADII = [72, 180]


def _reset_output_dir(output_dir):
    # limpa a pasta de saida pra nao misturar execucoes antigas
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _resize_bilinear(img, size):
    """Redimensiona uma imagem 2D pra size x size usando interpolacao bilinear.

    Fiz na mao pra nao depender de cv2.resize ou scipy. A interpolacao bilinear
    pondera os 4 vizinhos mais proximos pela distancia, gerando transicoes mais
    suaves que o nearest neighbor.
    """
    img = np.asarray(img, dtype=float)

    if img.ndim != 2:
        raise ValueError("A imagem deve ser 2D.")

    size = int(size)
    if size <= 0:
        raise ValueError("size deve ser positivo.")

    h, w = img.shape

    if h == size and w == size:
        return img.copy()

    row_coords = np.linspace(0, h - 1, size)
    col_coords = np.linspace(0, w - 1, size)

    r0 = np.floor(row_coords).astype(int)
    c0 = np.floor(col_coords).astype(int)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)

    dr = (row_coords - r0).reshape((size, 1))
    dc = (col_coords - c0).reshape((1, size))

    top_left = img[r0[:, None], c0[None, :]]
    top_right = img[r0[:, None], c1[None, :]]
    bottom_left = img[r1[:, None], c0[None, :]]
    bottom_right = img[r1[:, None], c1[None, :]]

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    return (1 - dr) * top + dr * bottom


def prepare_image_for_dft(path, size=DEFAULT_SIZE):
    # carrega, centraliza, redimensiona e normaliza uma imagem pra DFT
    img = read_gray(path).astype(float)
    square_size = min(img.shape[0], img.shape[1])
    cropped = center_crop(img, square_size)
    resized = _resize_bilinear(cropped, size)

    # mantenho a escala 0-255 em float pra facilitar comparacao visual e erro
    return normalize_0_255(resized).astype(float)


def _save_reconstruction_grid(output_path, original, radii, reconstructions):
    """Salva uma grade com a imagem original e reconstrucoes por raio.

    Cada raio representa quantas frequencias baixas foram mantidas. Quanto
    maior o raio, mais frequencias passam e mais a reconstrucao se aproxima
    da imagem original. Em aula vimos que as baixas frequencias carregam a
    estrutura geral e as altas frequencias carregam os detalhes finos.
    """
    images = [original] + list(reconstructions)
    titles = ["original"] + [f"raio={radius}" for radius in radii]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(3 * cols, 3))
    axes = np.asarray(axes).reshape(-1)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle("Reconstrucoes parciais com baixas frequencias")
    fig.tight_layout(pad=0.8)
    fig.savefig(output_path, bbox_inches="tight", dpi=220)
    plt.close(fig)


def _run_single_image_experiment(name, image_path, output_dir, size, radii):
    # executa DFT, espectro e reconstrucoes pra uma imagem
    image_dir = output_dir / name
    image_dir.mkdir(parents=True, exist_ok=True)

    img = prepare_image_for_dft(image_path, size=size)
    F = dft2_manual(img)
    spectrum = magnitude_spectrum(F)
    reconstructions = reconstruction_sequence(F, radii)
    reconstructions_by_radius = dict(zip(radii, reconstructions))

    original_path = image_dir / "00_imagem_usada.png"
    spectrum_path = image_dir / "01_espectro_magnitude.png"
    grid_path = image_dir / "02_reconstrucoes_progressivas.png"
    save_image(original_path, img)
    save_image(spectrum_path, spectrum)
    _save_reconstruction_grid(grid_path, img, radii, reconstructions)

    chosen_paths = []
    for index, radius in enumerate(CHOSEN_RADII, start=3):
        reconstruction = reconstructions_by_radius.get(radius)

        if reconstruction is None:
            reconstruction = partial_reconstruction(F, radius)

        partial_path = image_dir / f"{index:02d}_reconstrucao_raio_{radius}.png"
        save_image(partial_path, reconstruction)
        chosen_paths.append(partial_path)

    return {
        "name": name,
        "image": image_path,
        "size": size,
        "radii": radii,
        "outputs": [original_path, spectrum_path, grid_path] + chosen_paths,
    }


def run_part2_experiments(size=DEFAULT_SIZE, radii=None):
    """Gera os experimentos principais da Parte II pra duas imagens.

    Escolhi uma imagem com bastante alta frequencia (quadra de basquete, com
    linhas e detalhes) e outra mais suave (museu, com gradientes e areas
    uniformes) pra comparar como a reconstrucao parcial se comporta em
    cada caso.
    """
    if radii is None:
        radii = DEFAULT_RADII

    output_dir = Path("outputs") / "parte2_fourier"
    _reset_output_dir(output_dir)

    experiments = [
        {
            "name": "alta_frequencia",
            "image": Path("imgs") / "quadra-basquete-tdgarden.jpg",
        },
        {
            "name": "baixa_frequencia",
            "image": Path("imgs") / "pessoas-museu-belas-artes-interno.jpg",
        },
    ]

    metadata = []
    for experiment in experiments:
        item = _run_single_image_experiment(
            experiment["name"],
            experiment["image"],
            output_dir,
            size,
            radii,
        )
        metadata.append(item)
        print(f"{item['name']}: {item['image']} -> {output_dir / item['name']}")

    return metadata


def validate_dft_reconstruction(size=32):
    """Verifica numericamente que IDFT(DFT(img)) reconstroi a imagem.

    Esse teste eh importante pra garantir que a implementacao manual da DFT
    esta correta. Se o erro medio e maximo forem proximos de zero (ordem de
    1e-10 ou menos), a transformada esta funcionando direitinho.
    """
    image_path = Path("imgs") / "pessoas-museu-belas-artes-interno.jpg"
    img = prepare_image_for_dft(image_path, size=size)
    F = dft2_manual(img)
    reconstructed = idft2_manual(F)
    error = np.abs(img - reconstructed)

    result = {
        "image": image_path,
        "size": size,
        "mean_error": float(error.mean()),
        "max_error": float(error.max()),
    }

    print(
        "Validacao DFT/IDFT: "
        f"size={size}, erro medio={result['mean_error']:.12e}, "
        f"erro maximo={result['max_error']:.12e}"
    )

    return result
