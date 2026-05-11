# VisionFilters
Exploring Image Processing through Filtering and Fourier Transforms

This repository contains practical experiments and algorithms focusing on spatial filtering and frequency domain analysis (Fourier Transforms)

## How to run

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

All output images are generated automatically inside `outputs/`.

## Project structure

```
VisionFilters/
├── imgs/                   input images
├── outputs/                generated figures (created by main.py)
│   ├── parte1_filtros/         main filter results
│   ├── parte1_padding/         padding mode comparison
│   ├── parte1_frequencias/     frequency domain analysis
│   ├── parte1_pipelines/       real-world pipeline examples
│   └── parte2_fourier/         DFT/IDFT partial reconstructions
├── src/
│   ├── convolution.py          manual 2D convolution
│   ├── kernels.py              kernel definitions
│   ├── filters_spatial.py      spatial filters
│   ├── image_utils.py          I/O and helpers
│   ├── frequency_analysis.py   frequency visualization (Part I)
│   ├── fourier_manual.py       manual DFT/IDFT (Part II)
│   ├── experiments_part1.py    Part I experiment runners
│   └── experiments_part2.py    Part II experiment runners
├── report/relatorio.md         report
├── requirements.txt
└── main.py
```

## What main.py runs

| Step | What it does | Output folder |
|------|-------------|---------------|
| 1/6 | DFT numerical validation (sanity check) | — |
| 2/6 | Spatial filter gallery | `parte1_filtros/` |
| 3/6 | Padding mode comparison | `parte1_padding/` |
| 4/6 | Frequency domain analysis | `parte1_frequencias/` |
| 5/6 | Real-world pipeline examples | `parte1_pipelines/` |
| 6/6 | DFT/IDFT with partial reconstruction | `parte2_fourier/` |
