"""Rotinas para gerar imagens da Parte I do relatorio.

Conceito:
Cada experimento deve produzir figuras com uma pergunta clara: efeito do filtro,
comparacao com filtros parecidos, impacto do padding e comportamento em
frequencia. Este arquivo vai centralizar essas execucoes.
"""

from pathlib import Path

from src.image_utils import ensure_output_dirs, list_images


PART1_OUTPUT_GROUPS = {
    "padding": "parte1_padding",
    "filters": "parte1_filtros",
    "frequencies": "parte1_frequencias",
}


def print_available_images(imgs_dir="imgs"):
    """Mostra as imagens de entrada encontradas no projeto."""
    paths = list_images(imgs_dir)

    if len(paths) == 0:
        print(f"Nenhuma imagem encontrada em {imgs_dir}.")
        return []

    print("Imagens disponiveis:")
    for index, path in enumerate(paths, start=1):
        print(f"{index:02d}. {path}")

    return paths


def setup_project_outputs(base_dir="outputs"):
    """Cria as pastas de saida esperadas para o trabalho."""
    ensure_output_dirs(base_dir)
    return Path(base_dir)


def run_filter_gallery():
    """Gera comparacoes visuais dos filtros obrigatorios da Parte I."""
    raise NotImplementedError("Implementar quando os filtros espaciais estiverem prontos.")


def run_padding_comparison():
    """Gera imagens comparando constant, edge, reflect e wrap nas bordas."""
    raise NotImplementedError("Implementar depois da convolucao manual com padding.")


def run_frequency_comparison():
    """Gera visualizacoes dos kernels e imagens no dominio das frequencias."""
    raise NotImplementedError("Implementar depois das respostas em frequencia.")


def run_sharpening_processes():
    """Gera figuras de nitidez por Laplace e por unsharp mask."""
    raise NotImplementedError("Implementar depois dos processos de nitidez.")


def run_creative_process():
    """Gera a figura do filtro criativo envolvendo convolucao."""
    raise NotImplementedError("Implementar depois do kernel de relevo/emboss.")


def run_part1_all():
    """Executa todos os experimentos da Parte I na ordem do relatorio."""
    raise NotImplementedError("Implementar ao final da Parte I.")


def run_part1_architecture_check():
    """Verifica se a arquitetura base esta pronta para os proximos passos."""
    setup_project_outputs()
    print_available_images()
    print("Arquitetura da Parte I pronta. Implementacoes entram nas proximas etapas.")
