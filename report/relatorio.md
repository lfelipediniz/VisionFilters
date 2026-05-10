# VisionFilters

## Parte I - Filtros convolucionais

Este relatorio sera preenchido junto com os experimentos.

Perguntas que as imagens geradas devem responder:

- Qual e o proposito do filtro?
- Como ele se compara com filtros semelhantes?
- Qual o impacto de diferentes paddings?
- Como ele se comporta no dominio das frequencias?
- Quais frequencias ele suprime ou realca?
- Onde aparece no dia a dia?

## Organizacao prevista

- `src/image_utils.py`: leitura, conversao, normalizacao, exibicao e salvamento.
- `src/convolution.py`: convolucao manual e tratamento de padding.
- `src/kernels.py`: geracao dos kernels obrigatorios.
- `src/filters_spatial.py`: aplicacao dos filtros e processos da Parte I.
- `src/fourier_manual.py`: visualizacoes e respostas no dominio das frequencias.
- `src/experiments.py`: geracao das imagens para o relatorio.
- `main.py`: ponto de entrada para executar verificacoes e experimentos.

## Arquitetura da Parte I

### Ideia geral

A arquitetura separa o trabalho em tres camadas:

1. Base manual: `src/convolution.py` calcula a convolucao pixel a pixel.
2. Definicao dos filtros: `src/kernels.py` cria as matrizes de pesos.
3. Experimentos e relatorio: `src/filters_spatial.py`, `src/fourier_manual.py` e `src/experiments.py` aplicam os filtros e geram figuras.

Essa separacao segue o estilo dos codigos em sala: funcoes pequenas, NumPy para matrizes, `imageio` para leitura/salvamento e `matplotlib` para visualizacao.

### Funcoes principais planejadas

- `luminosity`, `norm_minmax`, `clip_uint8`, `read_gray`, `save_image`, `show_bw`: suporte de imagem.
- `validate_kernel`, `pad_image`, `conv_op`, `convolve_manual`: nucleo da convolucao manual.
- `shift_kernel`, `box_kernel`, `gaussian_kernel`, `laplace_kernel`, `sobel_kernels`, `emboss_kernel`: kernels da Parte I.
- `apply_shift`, `apply_box_blur`, `apply_gaussian_blur`, `apply_laplace`, `apply_sobel`: filtros obrigatorios.
- `sharpen_laplace`, `unsharp_mask`, `creative_convolution_process`: processos obrigatorios.
- `magnitude_spectrum`, `pad_kernel_to_shape`, `kernel_frequency_response`: visualizacao em frequencia.
- `run_filter_gallery`, `run_padding_comparison`, `run_frequency_comparison`, `run_sharpening_processes`, `run_creative_process`, `run_part1_all`: geracao das imagens finais.

### Imagens recomendadas

| Filtro/processo | Imagem recomendada | Motivo |
| --- | --- | --- |
| Shift | `pessoa-consulado-segurando-papel.jpg` | Fundo com tijolos e texto facilita perceber deslocamento. |
| Caixa/media | `pessoas-reuniao-cafe.jpg` | Rostos e mesa mostram suavizacao de detalhes finos. |
| Gaussiano | `pessoas-museu-belas-artes-interno.jpg` | Luz suave e rostos permitem comparar blur natural contra media. |
| Laplace | `pessoa-lado-estatua.jpg` | Pedra, janelas e escultura geram muitas transicoes de intensidade. |
| Sobel | `pessoa-estadio-futebol.jpg` | Placar, arquibancada e contornos do corpo mostram bordas claras. |
| Nitidez por Laplace | `caveira.jpg` | Texturas fortes mostram bem o realce de detalhes. |
| Unsharp mask | `pessoa-lado-bruxa-salem.jpg` | Contornos da estatua, galhos e roupa ajudam a ver nitidez sem exagero. |
| Padding | `pessoas-reuniao-cafe.jpg` | As faixas pretas nas bordas tornam o impacto do preenchimento evidente. |
| Frequencias | `quadra-basquete-tdgarden.jpg` | Telas, luzes e multidao combinam baixas e altas frequencias. |
| Criativo emboss | `pessoa-lado-estatua.jpg` | Relevo fica intuitivo em pedra, janelas e estatua. |

### Filtro criativo escolhido

O filtro criativo recomendado e o relevo/emboss. Ele usa convolucao com um kernel direcional, criando uma aparencia de iluminacao lateral. E uma boa escolha porque:

- e facil de explicar no relatorio;
- mostra claramente que a posicao dos pesos no kernel importa;
- se conecta naturalmente a bordas e altas frequencias;
- aparece no dia a dia em efeitos de edicao de foto, textura, impressao e design grafico.

### Pastas de saida

- `outputs/parte1_filtros/`: comparacoes dos filtros e processos.
- `outputs/parte1_padding/`: comparacoes de borda/padding.
- `outputs/parte1_frequencias/`: espectros e respostas em frequencia.
- `outputs/parte2_fourier/`: reservado para a Parte II, sem uso nesta etapa.
