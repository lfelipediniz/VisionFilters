# experimentos especificos da Parte I do trabalho
#
# aqui ficam as execucoes direcionadas pra gerar as figuras do relatorio.
# tem comparacao de paddings, galeria de filtros, pipelines do dia a dia
# e analise em frequencia de cada filtro

from pathlib import Path
import shutil

import numpy as np

from src.convolution import convolve2d
from src.filters_spatial import (
    apply_box_blur,
    apply_emboss,
    apply_gaussian_blur,
    apply_laplace,
    apply_shift,
    apply_sobel,
    sharpen_with_laplace,
    unsharp_mask,
)
from src.frequency_analysis import (
    combined_kernel_frequency_response,
    save_frequency_analysis_figure,
    save_frequency_analysis_figure_from_response,
)
from src.image_utils import read_gray, save_comparison_figure, save_image, to_uint8
from src.kernels import (
    box_kernel,
    emboss_kernel,
    gaussian_kernel,
    identity_kernel,
    laplace_kernel,
    shift_kernel,
    sobel_i_kernel,
    sobel_j_kernel,
)


PADDING_MODES = ["none", "zero", "edge", "reflect", "wrap"]
DEFAULT_PADDING = "reflect"


def _reset_output_dir(output_dir):
    # remove resultados antigos pra nao misturar imagens de execucoes anteriores
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in output_dir.iterdir():
        if path.name == ".gitkeep":
            continue

        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def _save_filter_comparison(output_path, original, result, filter_name, params):
    # salva original e resultado filtrado numa figura comparativa
    save_comparison_figure(
        output_path,
        [original, result],
        titles=["original", params],
        cols=2,
        figsize=(8, 4),
        main_title=filter_name,
    )


def _top_left_crop(img, size):
    # recorta o canto superior esquerdo pra destacar artefatos de borda
    img = np.asarray(img)
    size = min(int(size), img.shape[0], img.shape[1])
    return img[:size, :size]


def _save_padding_figure(output_path, original, results, filter_name, params, crop_size=None):
    # salva uma comparacao dos modos de padding pra um filtro
    images = [original] + [results[padding] for padding in PADDING_MODES]
    titles = ["original"] + [f"padding={padding}" for padding in PADDING_MODES]

    if crop_size is None:
        main_title = f"Padding - {filter_name}\n{params}"
    else:
        images = [_top_left_crop(image, crop_size) for image in images]
        main_title = f"Padding - {filter_name} - crop do canto superior esquerdo\n{params}"

    save_comparison_figure(
        output_path,
        images,
        titles=titles,
        cols=len(images),
        figsize=(18, 4),
        main_title=main_title,
        dpi=240,
        interpolation="nearest",
    )


def run_padding_experiment():
    """Gera figuras comparando paddings da Parte I.

    Escolhi a imagem pessoa-lado-bruxa-salem.jpg porque tem fundo claro, arvores
    finas e objetos encostando nas bordas, o que torna os artefatos de padding
    mais faceis de perceber. Em aula vimos que o modo de tratamento das bordas
    pode gerar artefatos visiveis, principalmente com kernels grandes.
    """
    image_path = Path("imgs") / "pessoa-lado-bruxa-salem.jpg"
    output_dir = Path("outputs") / "parte1_padding"
    _reset_output_dir(output_dir)
    full_dir = output_dir / "full"
    crops_dir = output_dir / "crops"
    full_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    # uso a imagem inteira pra preservar a composicao escolhida
    img = read_gray(image_path)
    crop_size = 220

    experiments = [
        {
            "filename": "01_shift.png",
            "filter_name": "Shift",
            "params": "kernel 31x31, di=15, dj=15",
            "apply": lambda image, padding: apply_shift(
                image,
                size=31,
                di=15,
                dj=15,
                padding=padding,
            ),
        },
        {
            "filename": "02_media_caixa.png",
            "filter_name": "Caixa/Media",
            "params": "kernel 15x15",
            "apply": lambda image, padding: apply_box_blur(
                image,
                size=15,
                padding=padding,
            ),
        },
        {
            "filename": "03_gaussiano.png",
            "filter_name": "Gaussiano",
            "params": "kernel 15x15, sigma=3.0",
            "apply": lambda image, padding: apply_gaussian_blur(
                image,
                size=15,
                sigma=3.0,
                padding=padding,
            ),
        },
        {
            "filename": "04_laplace.png",
            "filter_name": "Laplace",
            "params": "kernel 3x3 centro=-8",
            "apply": lambda image, padding: apply_laplace(image, padding=padding),
        },
        {
            "filename": "05_sobel.png",
            "filter_name": "Sobel magnitude",
            "params": "magnitude sqrt(Gi^2 + Gj^2)",
            "apply": lambda image, padding: apply_sobel(image, padding=padding),
        },
        {
            "filename": "06_unsharp_mask.png",
            "filter_name": "Unsharp Mask",
            "params": "gaussiano 9x9, sigma=2.0, amount=1.3",
            "apply": lambda image, padding: unsharp_mask(
                image,
                size=9,
                sigma=2.0,
                amount=1.3,
                padding=padding,
            ),
        },
    ]

    saved_paths = []

    for experiment in experiments:
        results = {}

        for padding in PADDING_MODES:
            results[padding] = experiment["apply"](img, padding)

        full_path = full_dir / experiment["filename"]
        crop_path = crops_dir / experiment["filename"]

        _save_padding_figure(
            full_path,
            img,
            results,
            experiment["filter_name"],
            experiment["params"],
        )
        _save_padding_figure(
            crop_path,
            img,
            results,
            experiment["filter_name"],
            experiment["params"],
            crop_size=crop_size,
        )

        saved_paths.extend([full_path, crop_path])
        print(f"{experiment['filter_name']}: {full_path} e {crop_path}")

    return saved_paths


def run_filter_experiments():
    # gera as figuras principais dos filtros e processos da Parte I
    output_dir = Path("outputs") / "parte1_filtros"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "filename": "01_shift.png",
            "filter_name": "Shift",
            "image": "pessoa-lado-bruxa-salem.jpg",
            "reason": "tem bordas, arvores e fundo claro, o que facilita perceber o deslocamento.",
            "params": "size=31, di=15, dj=15, padding=zero",
            "apply": lambda image: apply_shift(
                image,
                size=31,
                di=15,
                dj=15,
                padding="zero",
            ),
        },
        {
            "filename": "02_media_caixa.png",
            "filter_name": "Caixa/Media",
            "image": "pessoa-consulado-segurando-papel.jpg",
            "reason": "tem texto, bandeiras, tijolos e contornos bons para comparar perda de detalhe.",
            "params": "size=9, padding=reflect",
            "apply": lambda image: apply_box_blur(
                image,
                size=9,
                padding=DEFAULT_PADDING,
            ),
        },
        {
            "filename": "03_gaussiano.png",
            "filter_name": "Gaussiano",
            "image": "pessoa-consulado-segurando-papel.jpg",
            "reason": "permite comparar suavizacao mantendo texto e contornos reconheciveis.",
            "params": "size=7, sigma=1.5, padding=reflect",
            "apply": lambda image: apply_gaussian_blur(
                image,
                size=7,
                sigma=1.5,
                padding=DEFAULT_PADDING,
            ),
        },
        {
            "filename": "04_laplace.png",
            "filter_name": "Laplace",
            "image": "pessoas-estadio-futebol.jpg",
            "reason": "tem arquibancadas, linhas, rostos e muitas transicoes de intensidade.",
            "params": "abs(kernel 3x3 centro=-8), padding=reflect",
            "apply": lambda image: apply_laplace(image, padding=DEFAULT_PADDING),
        },
        {
            "filename": "05_sobel.png",
            "filter_name": "Sobel",
            "image": "pessoas-estadio-futebol.jpg",
            "reason": "tem muitas bordas direcionais nas arquibancadas, grades, campo e pessoas.",
            "params": "magnitude sqrt(Gi^2 + Gj^2), padding=reflect",
            "apply": lambda image: apply_sobel(image, padding=DEFAULT_PADDING),
        },
        {
            "filename": "06_nitidez_laplace.png",
            "filter_name": "Aumento de Nitidez com Laplace",
            "image": "pessoa-consulado-segurando-papel.jpg",
            "reason": "texto, tijolos e bandeiras deixam o ganho de nitidez bem visivel.",
            "params": "alpha=0.35, padding=reflect",
            "apply": lambda image: sharpen_with_laplace(
                image,
                alpha=0.35,
                padding=DEFAULT_PADDING,
            ),
        },
        {
            "filename": "07_unsharp_mask.png",
            "filter_name": "Unsharp Mask",
            "image": "pessoa-consulado-segurando-papel.jpg",
            "reason": "tem detalhes finos e texto, bons para ver a mascara de detalhe voltando para a imagem.",
            "params": "gaussiano size=7, sigma=1.5, amount=1.2, padding=reflect",
            "apply": lambda image: unsharp_mask(
                image,
                size=7,
                sigma=1.5,
                amount=1.2,
                padding=DEFAULT_PADDING,
            ),
        },
        {
            "filename": "08_emboss.png",
            "filter_name": "Emboss",
            "image": "caveira.jpg",
            "reason": "tem contraste e textura forte, ideais para o efeito criativo de relevo.",
            "params": "kernel emboss 3x3, padding=reflect",
            "apply": lambda image: apply_emboss(image, padding=DEFAULT_PADDING),
        },
    ]

    metadata = []

    for experiment in experiments:
        image_path = Path("imgs") / experiment["image"]
        original = read_gray(image_path)
        result = experiment["apply"](original)

        output_path = output_dir / experiment["filename"]
        _save_filter_comparison(
            output_path,
            original,
            result,
            experiment["filter_name"],
            experiment["params"],
        )

        metadata.append({
            "filter": experiment["filter_name"],
            "image": image_path,
            "reason": experiment["reason"],
            "output": output_path,
        })

    return metadata


def _apply_signed_kernel_for_view(img, kernel, padding=DEFAULT_PADDING):
    # aplica um kernel assinado e normaliza o modulo pra visualizacao
    response = convolve2d(img, kernel, padding=padding)
    return to_uint8(np.abs(response), normalize=True)


def _laplace_sharpening_kernel(alpha):
    # kernel equivalente de imagem_original - alpha * Laplace
    return identity_kernel(3) - alpha * laplace_kernel()


def _unsharp_mask_kernel(size, sigma, amount):
    # kernel equivalente de imagem + amount * (imagem - gaussiana)
    return (1 + amount) * identity_kernel(size) - amount * gaussian_kernel(size, sigma)


def _add_gaussian_noise(img, sigma=22.0, seed=7):
    # adiciona ruido gaussiano sintetico de forma reprodutivel
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=img.shape)
    return to_uint8(np.asarray(img, dtype=float) + noise)


def _save_daily_pipeline_figure(output_path, images, titles, main_title):
    # salva uma figura comparativa pros pipelines do dia a dia
    save_comparison_figure(
        output_path,
        images,
        titles=titles,
        cols=len(images),
        figsize=(4 * len(images), 4),
        main_title=main_title,
        dpi=240,
        interpolation="nearest",
    )


def run_daily_life_pipeline_experiments():
    """Gera pipelines cotidianos que combinam filtros da Parte I.

    Esses exemplos nao substituem os filtros obrigatorios. Eles mostram como
    filtros convolucionais aparecem em tarefas praticas do dia a dia, como
    reducao de ruido, pre-processamento pra bordas e combinacoes do tipo
    suavizar + derivar. Em aula vimos que suavizar antes de derivar eh
    fundamental pra evitar que o ruido domine a resposta.
    """
    output_dir = Path("outputs") / "parte1_pipelines"
    _reset_output_dir(output_dir)

    # imagem com texto, bandeiras, tijolos e contornos, boa pra perceber ruido,
    # perda de detalhe e diferenca entre bordas limpas e bordas contaminadas
    img = read_gray(Path("imgs") / "pessoa-consulado-segurando-papel.jpg")

    # simula ruido de sensor, aquelas variacoes aleatorias de intensidade que
    # aparecem em camera de celular com pouca luz ou ISO alto
    noisy = _add_gaussian_noise(img, sigma=22.0, seed=42)

    # 1) reducao de ruido com filtros de suavizacao. eles reduzem variacoes
    # aleatorias mas tambem podem apagar detalhes finos
    box_denoised = apply_box_blur(noisy, size=5, padding=DEFAULT_PADDING)
    gaussian_denoised = apply_gaussian_blur(
        noisy,
        size=7,
        sigma=1.4,
        padding=DEFAULT_PADDING,
    )
    denoise_path = output_dir / "01_reducao_ruido.png"
    _save_daily_pipeline_figure(
        denoise_path,
        [img, noisy, box_denoised, gaussian_denoised],
        [
            "original",
            "com ruido gaussiano",
            "caixa 5x5",
            "gaussiano 7x7 sigma=1.4",
        ],
        "Pipeline do dia a dia - reducao de ruido\nsimula ruido de sensor em cameras/celulares",
    )

    # 2) deteccao de bordas robusta. derivadas como Sobel amplificam ruido,
    # entao suavizar antes reduz bordas falsas. essa eh a ideia por tras de
    # detectores como Canny que a gente viu em aula
    sobel_noisy = apply_sobel(noisy, padding=DEFAULT_PADDING)
    smoothed_for_sobel = apply_gaussian_blur(
        noisy,
        size=7,
        sigma=1.4,
        padding=DEFAULT_PADDING,
    )
    gaussian_sobel = apply_sobel(smoothed_for_sobel, padding=DEFAULT_PADDING)
    sobel_path = output_dir / "02_bordas_robustas_sobel.png"
    _save_daily_pipeline_figure(
        sobel_path,
        [noisy, sobel_noisy, smoothed_for_sobel, gaussian_sobel],
        [
            "imagem ruidosa",
            "Sobel direto",
            "gaussiano antes",
            "Gaussiano + Sobel",
        ],
        "Pipeline do dia a dia - deteccao de bordas robusta\nderivadas amplificam ruido; suavizacao reduz bordas falsas",
    )

    # 3) Laplaciano de Gaussiano simplificado. o Laplace usa segunda derivada
    # e eh ainda mais sensivel a ruido. suavizar antes deixa as bordas mais limpas
    laplace_noisy = apply_laplace(noisy, padding=DEFAULT_PADDING)
    smoothed_for_laplace = apply_gaussian_blur(
        noisy,
        size=7,
        sigma=1.4,
        padding=DEFAULT_PADDING,
    )
    gaussian_laplace = apply_laplace(smoothed_for_laplace, padding=DEFAULT_PADDING)
    log_path = output_dir / "03_laplaciano_gaussiano_simplificado.png"
    _save_daily_pipeline_figure(
        log_path,
        [noisy, laplace_noisy, smoothed_for_laplace, gaussian_laplace],
        [
            "imagem ruidosa",
            "Laplace direto",
            "gaussiano antes",
            "Gaussiano + Laplace",
        ],
        "Pipeline do dia a dia - Laplaciano de Gaussiano simplificado\nLaplace e sensivel a ruido; suavizacao gera bordas mais limpas",
    )

    metadata = [
        {
            "pipeline": "Reducao de ruido",
            "output": denoise_path,
            "summary": "simula ruido de sensor e compara suavizacao por caixa e gaussiano.",
        },
        {
            "pipeline": "Deteccao de bordas robusta",
            "output": sobel_path,
            "summary": "mostra que Sobel direto amplifica ruido e que Gaussiano antes reduz bordas falsas.",
        },
        {
            "pipeline": "Laplaciano de Gaussiano simplificado",
            "output": log_path,
            "summary": "mostra que suavizar antes do Laplace gera resposta menos contaminada por ruido.",
        },
    ]

    for item in metadata:
        print(f"{item['pipeline']}: {item['output']}")

    return metadata


def run_frequency_experiments():
    """Gera analises em frequencia dos filtros e processos da Parte I.

    As imagens filtradas continuam sendo geradas pelos filtros espaciais com
    convolucao manual. O uso de np.fft fica em frequency_analysis.py apenas pra
    visualizar espectros e respostas dos kernels no relatorio. Em aula vimos
    que analisar o filtro no dominio da frequencia ajuda a entender se ele eh
    passa-baixa, passa-alta ou passa-banda.
    """
    output_dir = Path("outputs") / "parte1_frequencias"
    _reset_output_dir(output_dir)

    images = {
        "shift": Path("imgs") / "pessoa-lado-bruxa-salem.jpg",
        "smooth": Path("imgs") / "pessoa-consulado-segurando-papel.jpg",
        "edges": Path("imgs") / "pessoas-estadio-futebol.jpg",
        "emboss": Path("imgs") / "caveira.jpg",
    }

    shift_size = 31
    shift_di = 15
    shift_dj = 15
    box_size = 9
    gaussian_size = 9
    gaussian_sigma = 1.8
    laplace_alpha = 0.35
    unsharp_size = 7
    unsharp_sigma = 1.5
    unsharp_amount = 1.2

    sobel_i = sobel_i_kernel()
    sobel_j = sobel_j_kernel()

    experiments = [
        {
            "filename": "01_shift.png",
            "filter_name": "Shift",
            "image": images["shift"],
            "kernel": shift_kernel(shift_size, shift_di, shift_dj),
            "params": (
                "kernel 31x31, di=15, dj=15, padding=wrap; "
                "shift altera principalmente a fase, entao a magnitude quase nao muda"
            ),
            "apply": lambda image: apply_shift(
                image,
                size=shift_size,
                di=shift_di,
                dj=shift_dj,
                padding="wrap",
            ),
            "summary": "magnitude quase plana; deslocamento muda fase, nao conteudo de frequencias.",
        },
        {
            "filename": "02_media_caixa.png",
            "filter_name": "Caixa/Media",
            "image": images["smooth"],
            "kernel": box_kernel(box_size),
            "params": (
                "size=9, padding=reflect; resposta lembra sinc 2D e atua como passa-baixa"
            ),
            "apply": lambda image: apply_box_blur(
                image,
                size=box_size,
                padding=DEFAULT_PADDING,
            ),
            "summary": "passa-baixa com lobos de sinc; suaviza detalhes finos.",
        },
        {
            "filename": "03_gaussiano.png",
            "filter_name": "Gaussiano",
            "image": images["smooth"],
            "kernel": gaussian_kernel(gaussian_size, gaussian_sigma),
            "params": (
                "size=9, sigma=1.8, padding=reflect; resposta suave concentrada no centro"
            ),
            "apply": lambda image: apply_gaussian_blur(
                image,
                size=gaussian_size,
                sigma=gaussian_sigma,
                padding=DEFAULT_PADDING,
            ),
            "summary": "passa-baixa suave; reduz altas frequencias sem os lobos fortes da media.",
        },
        {
            "filename": "04_laplace.png",
            "filter_name": "Laplace",
            "image": images["edges"],
            "kernel": laplace_kernel(),
            "params": (
                "kernel 3x3 centro=-8, padding=reflect; centro suprimido e altas frequencias realcadas"
            ),
            "apply": lambda image: apply_laplace(image, padding=DEFAULT_PADDING),
            "summary": "passa-alta; responde pouco a regioes uniformes e muito a bordas.",
        },
        {
            "filename": "05_sobel_horizontal.png",
            "filter_name": "Sobel i - bordas horizontais",
            "image": images["edges"],
            "kernel": sobel_i,
            "params": (
                "Sobel i, padding=reflect; resposta direcional pra variacoes no eixo i"
            ),
            "apply": lambda image: _apply_signed_kernel_for_view(
                image,
                sobel_i,
                padding=DEFAULT_PADDING,
            ),
            "summary": "passa-alta direcional; enfatiza bordas horizontais.",
        },
        {
            "filename": "06_sobel_vertical.png",
            "filter_name": "Sobel j - bordas verticais",
            "image": images["edges"],
            "kernel": sobel_j,
            "params": (
                "Sobel j, padding=reflect; resposta direcional pra variacoes no eixo j"
            ),
            "apply": lambda image: _apply_signed_kernel_for_view(
                image,
                sobel_j,
                padding=DEFAULT_PADDING,
            ),
            "summary": "passa-alta direcional; enfatiza bordas verticais.",
        },
        {
            "filename": "07_sobel_magnitude.png",
            "filter_name": "Sobel magnitude",
            "image": images["edges"],
            "response": lambda shape: combined_kernel_frequency_response(
                [sobel_i, sobel_j],
                shape,
            ),
            "params": (
                "sqrt(Gi^2 + Gj^2), padding=reflect; nao eh um unico kernel, combina Sobel i e j"
            ),
            "response_title": "resposta combinada |Sobel|",
            "apply": lambda image: apply_sobel(image, padding=DEFAULT_PADDING),
            "summary": "combina duas direcoes; realca bordas em varias orientacoes.",
        },
        {
            "filename": "08_nitidez_laplace.png",
            "filter_name": "Aumento de nitidez com Laplace",
            "image": images["smooth"],
            "kernel": _laplace_sharpening_kernel(laplace_alpha),
            "params": (
                "imagem - 0.35 * Laplace; preserva a imagem e amplifica altas frequencias"
            ),
            "apply": lambda image: sharpen_with_laplace(
                image,
                alpha=laplace_alpha,
                padding=DEFAULT_PADDING,
            ),
            "summary": "mantem baixas frequencias da imagem original e reforca detalhes.",
        },
        {
            "filename": "09_unsharp_mask.png",
            "filter_name": "Unsharp Mask",
            "image": images["smooth"],
            "kernel": _unsharp_mask_kernel(unsharp_size, unsharp_sigma, unsharp_amount),
            "params": (
                "gaussiano size=7, sigma=1.5, amount=1.2; altas frequencias aumentadas suavemente"
            ),
            "apply": lambda image: unsharp_mask(
                image,
                size=unsharp_size,
                sigma=unsharp_sigma,
                amount=unsharp_amount,
                padding=DEFAULT_PADDING,
            ),
            "summary": "preserva estrutura geral e devolve parte dos detalhes removidos pelo blur.",
        },
        {
            "filename": "10_emboss.png",
            "filter_name": "Emboss",
            "image": images["emboss"],
            "kernel": emboss_kernel(),
            "params": (
                "kernel emboss 3x3, padding=reflect; passa-alta direcional com aparencia de relevo"
            ),
            "apply": lambda image: apply_emboss(image, padding=DEFAULT_PADDING),
            "summary": "passa-alta direcional; bordas ganham efeito de iluminacao lateral.",
        },
    ]

    metadata = []

    for experiment in experiments:
        original = read_gray(experiment["image"])
        filtered = experiment["apply"](original)
        output_path = output_dir / experiment["filename"]

        if "kernel" in experiment:
            save_frequency_analysis_figure(
                output_path,
                original,
                experiment["kernel"],
                filtered,
                experiment["filter_name"],
                params=experiment["params"],
            )
        else:
            response_img = experiment["response"](original.shape)
            save_frequency_analysis_figure_from_response(
                output_path,
                original,
                response_img,
                filtered,
                experiment["filter_name"],
                params=experiment["params"],
                response_title=experiment["response_title"],
            )

        metadata.append({
            "filter": experiment["filter_name"],
            "image": experiment["image"],
            "output": output_path,
            "summary": experiment["summary"],
        })

        print(f"{experiment['filter_name']}: {output_path}")

    return metadata
