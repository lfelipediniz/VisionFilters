"""Ponto de entrada do trabalho VisionFilters.

Neste momento o main apenas verifica a arquitetura. Nos proximos passos ele
podera chamar os experimentos que geram as imagens do relatorio.
"""

from src.experiments import run_part1_architecture_check


def main():
    run_part1_architecture_check()


if __name__ == "__main__":
    main()

