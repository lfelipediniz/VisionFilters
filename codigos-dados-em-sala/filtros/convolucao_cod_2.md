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
def norm_minmax(img, C=255, m=0):
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
Filtro de Shift

def shift_kernel(k):

    kernel = np.zeros((k, k))
    kernel[k - 1, k - 1] = 1

    return kernel

show_bw(convolve(img, shift_kernel(49)))
No description has been provided for this image
Filtro de Caixa/Média

def box_kernel(size):
    return np.ones((size, size)) / float(size**2)

show_bw(convolve(img, box_kernel(13)))
No description has been provided for this image
Filtro Gaussiano

G
2
D
(
x
,
y
,
σ
)
=
1
2
π
σ
2
e
−
x
2
+
y
2
2
σ
2

def gaussian_kernel(size, sigma):
    a = b = size // 2
    i_dir = np.arange(-1, 0.99, 2.0 / size)
    j_dir = np.arange(-1, 0.99, 2.0 / size)
    
    kernel = np.zeros((size, size))
    for i, i_sample in enumerate(i_dir):
        for j, j_sample in enumerate(j_dir):
            kernel[i, j] = np.exp(-(i_sample**2 + j_sample**2) / (2 * (sigma**2)))
    kernel = kernel / kernel.sum()
    return kernel

show_bw(convolve(img, gaussian_kernel(5, 2)))
No description has been provided for this image
Filtro de Mediana

def median_filter(image, k):
    new_img = np.zeros_like(image)
    h, w = image.shape

    a = b = k // 2

    img_pad = np.pad(image, a, mode='reflect')

    for i in range(0, h):
        for j in range(0, w):
            neighbourhood = img_pad[i:i + 2 * a + 1, j:j + 2 * b + 1]
            new_img[i, j] = np.median(neighbourhood)

    return new_img

show_bw(median_filter(img, 9))
No description has been provided for this image
Filtro de Laplace e Aumento de Nitidez

def laplace():
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])

def sharpen(img, alpha=0.1):
    nimg = convolve(img, laplace())
    return norm_minmax(np.absolute((img.astype(float) - alpha*nimg))).astype(np.uint8)

img = iio.imread('moon.png')
img = luminosity(img)
show_bw(convolve(img, laplace()))
No description has been provided for this image
show_bw(sharpen(img, 0.5))
No description has been provided for this image
Filtro com Máscara de Unsharpening ("des-nitidez?")

Filtro de Sobel

def sobel_j_kernel():
    return np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
def sobel_i_kernel():
    return np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]])