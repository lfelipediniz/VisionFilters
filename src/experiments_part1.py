"""Experimentos especificos da Parte I do trabalho.

Este arquivo guarda execucoes pequenas e direcionadas para gerar figuras do
relatorio. A primeira delas compara como diferentes paddings afetam filtros
convolucionais.
"""

from pathlib import Path
import shutil

from src.filters_spatial import apply_box_blur, apply_shift, apply_sobel
from src.image_utils import read_gray, save_comparison_figure


PADDING_MODES = ["none", "zero", "edge", "reflect", "wrap"]


def _run_filter_for_all_paddings(img, filter_function):
    """Aplica o mesmo filtro em todos os modos de padding planejados."""
    results = []

    for padding in PADDING_MODES:
        results.append(filter_function(img, padding))

    return results


def _save_padding_comparison(output_path, original, results, filter_name):
    """Salva uma figura com original + resultados lado a lado."""
    images = [original] + results
    titles = ["original"] + PADDING_MODES

    save_comparison_figure(
        output_path,
        images,
        titles=titles,
        cols=len(images),
        figsize=(18, 4),
        main_title=f"Padding - {filter_name}",
    )


def run_padding_experiment():
    """Gera figuras comparando paddings para filtros da Parte I.

    A imagem escolhida foi `pessoa-lado-bruxa-salem.jpg`, porque tem fundo
    claro, arvores finas e objetos encostando em regioes proximas das bordas.
    Isso torna os artefatos de padding mais faceis de perceber.
    """
    image_path = Path("imgs") / "pessoa-lado-bruxa-salem.jpg"
    output_dir = Path("outputs") / "parte1" / "padding"
    legacy_output_dir = Path("outputs") / "parte1_padding"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_output_dir.mkdir(parents=True, exist_ok=True)

    # As imagens de entrada ja foram preparadas em formato quadrado, entao
    # usamos a imagem inteira para preservar a composicao escolhida.
    img = read_gray(image_path)

    experiments = [
        (
            "media_9x9",
            "media 9x9",
            lambda image, padding: apply_box_blur(image, size=9, padding=padding),
        ),
        (
            "sobel",
            "Sobel",
            lambda image, padding: apply_sobel(image, padding=padding),
        ),
        (
            "shift_di5_dj5",
            "shift di=5 dj=5",
            lambda image, padding: apply_shift(image, size=11, di=5, dj=5, padding=padding),
        ),
    ]

    saved_paths = []

    for filename, filter_name, filter_function in experiments:
        results = _run_filter_for_all_paddings(img, filter_function)
        output_path = output_dir / f"padding_{filename}.png"
        _save_padding_comparison(output_path, img, results, filter_name)
        shutil.copy2(output_path, legacy_output_dir / output_path.name)
        saved_paths.append(output_path)

    return saved_paths
