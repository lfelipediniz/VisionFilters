Processamento de Imagens
Filtros e Convolução
Professora Leo Sampaio Ferraz Ribeiro

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
Funções de Suporte

# função de luminosidade para converter imagens coloridas para preto&branco
def luminosity(img):
    return (0.21*img[..., 0] + 0.72*img[..., 1] + 0.07*img[..., 2]).astype(np.uint8)

# normalização min-max para sempre deixar imagens em [0-255]
def norm_minmax(img, C, m):
    img = (img - img.min())/(img.max() - img.min())
    img *= C
    img -= m
    return img

# professora vai esquecer do cmap='gray' toda vez que quiser mostrar uma imagem
def show_bw(img):
    plt.imshow(img, cmap='gray')
Função de convolução
def conv_op(i, j, img, kernel):

    # como não conseguimos definir um kernel centralizado em índice 0
    # calculamos o deslocamento nas duas direções como a e b
    k_h, k_w = kernel.shape
    a = (k_h-1) // 2
    b = (k_w-1) // 2

    # kernel fica centralizado sobre (i, j) na imagem
    # selecionamos a região em torno de (i, j)
    neighbourhood = img[i-a:i+a+1, j-b:j+b+1]

    # multiplicação ponto a ponto e somatório no final
    c_mul = kernel*neighbourhood
    return c_mul.sum()

def convolve(img, kernel):

    img = img.astype(float)
    new_img = np.zeros_like(img)
    h, w = img.shape

    kernel = np.flip(kernel)
    k_h, k_w = kernel.shape
    a = (k_h-1) // 2
    b = (k_w-1) // 2

    # como o kernel precisa estar centralizado em cada pixel, 
    # consideramos aqui que não vamos trabalhar nos cantos
    for i in range(a, h-a):
        for j in range(b, w-b):
            new_img[i, j] = conv_op(i, j, img, kernel)

    return new_img
Definição de Filtros e Aplicação
img = iio.imread('XT206484.jpeg')
img = luminosity(img)
show_bw(img)
No description has been provided for this image
Filtro de média

kernel = np.ones((7, 7))/49
new_img = convolve(img, kernel)
show_bw(new_img)
No description has been provided for this image
Filtro de Bordas (sobel)

kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
new_img = convolve(img, kernel)
show_bw(new_img)
No description has been provided for this image
Combinação de Média e Sobel

kernel_mean = np.ones((7, 7))/49
kernel_sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
new_img = convolve(convolve(img, kernel_mean), kernel_sobel)
show_bw(new_img)
No description has been provided for this image
 