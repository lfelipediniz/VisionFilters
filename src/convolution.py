# convolucao manual em 2D
#
# a convolucao basicamente posiciona um kernel sobre cada pixel, multiplica os
# valores da vizinhanca pelos pesos do kernel e soma tudo pra gerar o novo pixel.
# aqui eu implemento tudo na mao, sem usar cv2.filter2D, scipy.signal.convolve2d
# nem ndimage.convolve, como foi pedido

import numpy as np


PADDING_ALIASES = {
    "none": "none",
    "zero": "zero",
    "constant": "zero",
    "edge": "edge",
    "reflect": "reflect",
    "wrap": "wrap",
}


def validate_kernel(kernel):
    # confere se o kernel eh bidimensional e tem dimensoes impares
    kernel = np.asarray(kernel, dtype=float)

    if kernel.ndim != 2:
        raise ValueError("O kernel deve ser uma matriz 2D.")

    k_h, k_w = kernel.shape
    if k_h % 2 == 0 or k_w % 2 == 0:
        raise ValueError("O kernel deve ter altura e largura impares.")

    return kernel


def _normalize_padding_name(mode):
    # padroniza os nomes de padding que o projeto aceita
    if mode not in PADDING_ALIASES:
        valid_modes = ", ".join(sorted(PADDING_ALIASES))
        raise ValueError(f"Padding invalido: {mode}. Use um destes: {valid_modes}.")

    return PADDING_ALIASES[mode]


def pad_image(img, pad_h, pad_w, mode="zero"):
    """Prepara a imagem pra tratar as bordas durante a convolucao.

    Em aula a gente viu que quando o kernel chega perto da borda, faltam vizinhos.
    Cada modo de padding resolve isso de um jeito diferente
    - none nao adiciona borda (o pixel simplesmente nao eh calculado)
    - zero completa com zeros ao redor
    - edge repete o pixel mais proximo da borda
    - reflect espelha a imagem na borda
    - wrap conecta lados opostos, como se a imagem fosse periodica
    """
    img = np.asarray(img)

    if img.ndim != 2:
        raise ValueError("A imagem deve ser 2D, em escala de cinza.")

    if pad_h < 0 or pad_w < 0:
        raise ValueError("Os tamanhos de padding devem ser nao negativos.")

    pad_h = int(pad_h)
    pad_w = int(pad_w)
    mode = _normalize_padding_name(mode)

    if mode == "none" or (pad_h == 0 and pad_w == 0):
        return img.copy()

    pad_width = ((pad_h, pad_h), (pad_w, pad_w))

    if mode == "zero":
        return np.pad(img, pad_width, mode="constant", constant_values=0)

    return np.pad(img, pad_width, mode=mode)


def conv_op(i, j, img, kernel):
    """Calcula um unico pixel filtrado a partir da vizinhanca de (i, j).

    Essa funcao assume que a imagem e o kernel ja foram validados por convolve2d.
    Como ela roda pra cada pixel, deixei aqui so o essencial pra nao pesar.
    """
    k_h, k_w = kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    i_start = i - a
    i_end = i + a + 1
    j_start = j - b
    j_end = j + b + 1

    if i_start < 0 or j_start < 0 or i_end > img.shape[0] or j_end > img.shape[1]:
        raise ValueError("A vizinhanca do pixel saiu dos limites da imagem.")

    # a vizinhanca tem exatamente o mesmo tamanho do kernel centralizado
    neighbourhood = img[i_start:i_end, j_start:j_end]

    # multiplicacao ponto a ponto seguida do somatorio, como a gente viu em aula
    return (kernel * neighbourhood).sum()


def convolve2d(img, kernel, padding="zero"):
    """Aplica convolucao 2D manual em uma imagem em escala de cinza.

    Antes de aplicar, o kernel eh invertido (flip) pra ser convolucao de verdade.
    Sem essa inversao a operacao seria correlacao, que eh parecida mas nao identica.
    Essa distincao entre convolucao e correlacao foi bastante enfatizada em aula.
    """
    img = np.asarray(img, dtype=float)
    kernel = validate_kernel(kernel)
    padding = _normalize_padding_name(padding)

    if img.ndim != 2:
        raise ValueError("A imagem deve ser 2D, em escala de cinza.")

    new_img = np.zeros_like(img, dtype=float)
    h, w = img.shape

    # np.flip inverte linhas e colunas, fazendo o flip 2D do kernel
    flipped_kernel = np.flip(kernel)
    k_h, k_w = flipped_kernel.shape
    a = (k_h - 1) // 2
    b = (k_w - 1) // 2

    if padding == "none":
        # sem padding so calcula os pixels cuja vizinhanca inteira existe
        # dentro da imagem original, entao a saida perde as bordas
        for i in range(a, h - a):
            for j in range(b, w - b):
                new_img[i, j] = conv_op(i, j, img, flipped_kernel)

        return new_img

    img_pad = pad_image(img, a, b, mode=padding)

    # na imagem com padding, o pixel original (i, j) fica em (i+a, j+b)
    for i in range(h):
        for j in range(w):
            new_img[i, j] = conv_op(i + a, j + b, img_pad, flipped_kernel)

    return new_img


def convolve_manual(img, kernel, padding="zero"):
    # alias da funcao central da Parte I, mantido pra clareza
    return convolve2d(img, kernel, padding=padding)
