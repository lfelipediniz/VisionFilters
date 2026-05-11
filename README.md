# VisionFilters

Image filtering and Fourier transform experiments.
Assignment for the Image Processing course at USP (2026).

## Requirements

Python 3.10+ and the following packages:

```
pip install numpy imageio matplotlib
```

## Project structure

```
VisionFilters/
├── imgs/               input images
├── outputs/            generated figures (created automatically)
│   ├── parte1_filtros/
│   ├── parte1_padding/
│   ├── parte1_frequencias/
│   ├── parte1_pipelines/
│   └── parte2_fourier/
├── src/
│   ├── convolution.py      manual 2D convolution
│   ├── kernels.py          kernel definitions
│   ├── filters_spatial.py  spatial filters
│   ├── image_utils.py      I/O and helpers
│   ├── frequency_analysis.py  frequency visualization (Part I)
│   ├── fourier_manual.py   manual DFT/IDFT (Part II)
│   ├── experiments_part1.py
│   └── experiments_part2.py
├── report/relatorio.md     report
└── main.py
```

## Running the experiments

**Part I - spatial filters and frequency analysis:**

```python
from src.experiments_part1 import (
    run_padding_experiment,
    run_filter_experiments,
    run_frequency_experiments,
    run_daily_life_pipeline_experiments,
)

run_filter_experiments()
run_padding_experiment()
run_frequency_experiments()
run_daily_life_pipeline_experiments()
```

**Part II - manual DFT/IDFT and partial reconstruction:**

```python
from src.experiments_part2 import run_part2_experiments, validate_dft_reconstruction

validate_dft_reconstruction(size=32)   # sanity check
run_part2_experiments()                # generates outputs/parte2_fourier/
```

All output images are saved automatically to the `outputs/` folder.
