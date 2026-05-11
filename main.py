# ponto de entrada do trabalho VisionFilters
# rode com: python main.py
# todos os resultados sao salvos automaticamente em outputs/

from src.experiments_part1 import (
    run_daily_life_pipeline_experiments,
    run_filter_experiments,
    run_frequency_experiments,
    run_padding_experiment,
)
from src.experiments_part2 import run_part2_experiments, validate_dft_reconstruction


def main():
    print("=== VisionFilters - gerando todos os outputs ===\n")

    print("[1/6] validando implementacao manual da DFT...")
    validate_dft_reconstruction(size=32)

    print("\n[2/6] Parte I - filtros espaciais (galeria principal)...")
    run_filter_experiments()

    print("\n[3/6] Parte I - comparacao de modos de padding...")
    run_padding_experiment()

    print("\n[4/6] Parte I - analise no dominio das frequencias...")
    run_frequency_experiments()

    print("\n[5/6] Parte I - pipelines do dia a dia...")
    run_daily_life_pipeline_experiments()

    print("\n[6/6] Parte II - DFT e IDFT manuais com reconstrucao parcial...")
    run_part2_experiments()

    print("\n=== pronto! todos os resultados estao em outputs/ ===")


if __name__ == "__main__":
    main()
