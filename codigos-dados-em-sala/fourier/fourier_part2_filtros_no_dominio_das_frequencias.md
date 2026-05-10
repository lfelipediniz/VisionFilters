Processamento de Imagens
Transformada de Fourier (Parte II)
Professora Leo Sampaio Ferraz Ribeiro

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
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

def show_bw2(ogimg, img):
    plt.figure(figsize=(10, 20))
    plt.subplot(1, 2, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')

def pad_filter(kernel, img):
    h, w = img.shape
    hk, wk = kernel.shape
    return np.pad(kernel, ((h//2 - hk//2, h//2 - hk//2),(w//2 - wk//2, w//2 - wk//2)))

def show_fft(ogimg, show_phase=False):
    plt.figure(figsize=(15, 30))
    f_img = np.log2(np.fft.fftshift(np.fft.fft2(ogimg))+0.0001).real
    plt.subplot(1, 2 if not show_phase else 3, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(1, 2 if not show_phase else 3, 2)
    plt.imshow(f_img, cmap='gray')
    if show_phase:
        f_img = np.fft.fft2(ogimg)
        plt.subplot(1, 3, 3)
        plt.imshow(np.arctan2(f_img.imag, f_img.real), cmap='gray')

def show_fft_filter(ogimg, kernel):
    plt.figure(figsize=(20, 10))
    f_img = np.fft.fft2(ogimg)
    plt.subplot(2, 3, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.imshow(np.log2(np.fft.fftshift(f_img)+0.0001).real, cmap='gray')
    kernel_pad = pad_filter(kernel, ogimg)
    f_kernel = np.fft.fft2(kernel_pad, s=ogimg.shape)
    plt.subplot(2, 3, 3)
    plt.imshow(kernel_pad, cmap='gray')
    plt.subplot(2, 3, 4)
    plt.imshow(np.log2(np.fft.fftshift(f_kernel)+0.0001).real, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow(np.log2(np.fft.fftshift(f_img*f_kernel)+0.0001).real, cmap='gray')
    plt.subplot(2, 3, 6)
    plt.imshow(np.fft.fftshift(np.fft.ifft2(f_img*f_kernel)).real, cmap='gray')

def show_fft_fd_filter(ogimg, kernel):
    plt.figure(figsize=(20, 10))
    f_img = np.fft.fft2(ogimg)
    plt.subplot(2, 3, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(2, 3, 2)
    plt.imshow(np.log2(np.fft.fftshift(f_img)+0.0001).real, cmap='gray')
    plt.subplot(2, 3, 3)
    plt.imshow(kernel, cmap='gray')
    plt.subplot(2, 3, 4)
    kernel = np.fft.fftshift(kernel)
    plt.imshow(np.log2(np.fft.fftshift(f_img*kernel)+0.0001).real, cmap='gray')
    plt.subplot(2, 3, 5)
    plt.imshow((np.fft.ifft2(f_img*kernel)).real, cmap='gray')
Efeitos na Imagem do Retân
def inv_rot_matrix(theta):
    return np.array([[np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]])
def inv_scale_matrix(si, sj):
    return np.array([[1.0 / si, 0, 0],
            [0, 1.0 / sj, 0],
            [0, 0, 1]] )
def inv_translation_matrix(ti, tj):
    return np.array([[1, 0, -ti],
            [0, 1, -tj],
            [0, 0, 1]])
def transform(img, theta, si, sj, ti, tj):
    mat_inv = inv_scale_matrix(si, sj)
    h, w = img.shape
    mat_inv = mat_inv@inv_translation_matrix(-h/2.0, -w/2.0)
    mat_inv = mat_inv@inv_rot_matrix(theta)
    mat_inv = mat_inv@inv_translation_matrix(h/2.0, w/2.0)
    mat_inv = mat_inv@inv_translation_matrix(ti, tj)
    new_img = np.zeros_like(img)
    for i in range(h):
        for j in range(w):
            p = np.array([i, j, 1])
            pl = mat_inv@p
    
            i_og = int(np.round(pl[0]))
            j_og = int(np.round(pl[1]))
    
            if not (i_og >= h or j_og >= w or i_og < 0 or j_og < 0):
                new_img[i, j] = img[i_og, j_og]
    return new_img
img_teste = np.zeros((64, 64))
img_teste[10:20, 10:20] = 255.
show_bw(img_teste)
No description has been provided for this image
Transformada de Fourier

show_fft(img_teste, show_phase=True)
No description has been provided for this image
Translação

show_fft(transform(img_teste, theta=0, si=1, sj=1, ti=20, tj=1), show_phase=True)
No description has been provided for this image
Rotação

show_fft(transform(img_teste, theta=np.pi/4, si=1, sj=1, ti=0, tj=0), show_phase=True)
No description has been provided for this image
Transformada em Imagens Simples
x_s = np.arange(0, 1, 1/128)

img_s = np.zeros((128, 128))
for i, x in enumerate(x_s):
    for j, y in enumerate(x_s):
        img_s[i, j] = np.sin(2*np.pi*5*x)
show_fft(img_s)
No description has been provided for this image
x_s = np.arange(0, 1, 1/128)

img_s = np.zeros((128, 128))
for i, x in enumerate(x_s):
    for j, y in enumerate(x_s):
        img_s[i, j] = np.sin(2*np.pi*50*x)
show_fft(img_s)
No description has been provided for this image
x_s = np.arange(0, 1, 1/128)

img_s = np.zeros((128, 128))
for i, x in enumerate(x_s):
    for j, y in enumerate(x_s):
        img_s[i, j] = np.sin(2*np.pi*50*x) + np.sin(2*np.pi*50*y)
show_fft(img_s)
No description has been provided for this image
x_s = np.arange(0, 1, 1/128)

img_s = np.zeros((128, 128))
for i, x in enumerate(x_s):
    for j, y in enumerate(x_s):
        img_s[i, j] = np.sin(2*np.pi*50*x) + np.sin(2*np.pi*50*y) + np.sin(2*np.pi*100*x) + np.sin(2*np.pi*28*y) + 0.5*np.sin(2*np.pi*10*y) + 0.2*np.sin(2*np.pi*1*x)
show_fft(img_s)
No description has been provided for this image
Observando imagens de baixa e alta frequência
img_h = iio.imread('bambu.jpeg')
img_h = luminosity(img_h)
show_fft(img_h)
No description has been provided for this image
img_h = iio.imread('caaso.jpeg')
img_h = luminosity(img_h)
show_fft(img_h)
No description has been provided for this image
img = iio.imread('pompom_brownie.jpeg')
img = luminosity(img)
show_fft(img)
No description has been provided for this image
Filtros no Domínio das Frequências
def pad_filter(kernel, img):
    h, w = img.shape
    hk, wk = kernel.shape
    return np.pad(kernel, ((h//2 - hk//2, h//2 - hk//2),(w//2 - wk//2, w//2 - wk//2)))
Filtro de Caixa

def mean_f(size):
    return np.ones((size, size))/(size**2)

show_fft(pad_filter(mean_f(5), img))
No description has been provided for this image
show_fft_filter(img, mean_f(5))
No description has been provided for this image
Filtros de Sobel

def sobel_i():
    return np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])
def sobel_j():
    return np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

show_fft_filter(img, sobel_i())
No description has been provided for this image
Filtro Gaussiano

def gaussian(size, sigma):
    x_s = np.linspace(-1, 1, size[0])
    y_s = np.linspace(-1, 1, size[1])
    kernel = np.zeros(size)
    for i, x in enumerate(x_s):
        for j, y in enumerate(y_s):
            kernel[i, j] = np.exp(-((x**2)/(2*sigma[0]**2)+(y**2)/(2*sigma[1]**2)))
            kernel[i, j] /= 2*np.pi*sigma[0]*sigma[1]
    return kernel/kernel.sum()
show_fft_filter(img, gaussian([5, 5], [0.5, 0.5]))
No description has been provided for this image
Filtro Gaussiano definido no domínio das frequências

show_fft_fd_filter(img, gaussian(img.shape, [0.3, 0.3]))
No description has been provided for this image
Filtro de Laplace

img_l = iio.imread('laplace.jpeg')
img_l = luminosity(img_l)
def laplace():
    return np.array([[1, 1, 1],
                     [1, -8, 1],
                     [1, 1, 1]])
show_fft_filter(img_l, laplace())
No description has been provided for this image
Filtro de Laplace definido no domínio das frequências

def laplacian_high_pass(size):
    P, Q = size
    filter_f = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            D = (u - P / 2)**2 + (v - Q / 2)**2
            filter_f[u, v] = (4) * (np.pi**2) * D
    return filter_f
show_fft_fd_filter(img_l, laplacian_high_pass(img_l.shape))
No description has been provided for this image
Filtros Perfeitos
def ideal_low_pass(size, r):
    P, Q = size
    filter_f = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            if D <= r:
                filter_f[u, v] = 1
    return filter_f
show_fft_fd_filter(img, ideal_low_pass(img.shape, 10))
No description has been provided for this image
def ideal_high_pass(size, r):
    P, Q = size
    filter_f = np.zeros((P, Q), dtype=np.float32)
    for u in range(P):
        for v in range(Q):
            D = np.sqrt((u - P / 2)**2 + (v - Q / 2)**2)
            if D >= r:
                filter_f[u, v] = 1
    return filter_f
show_fft_fd_filter(img, ideal_high_pass(img.shape, 10))
No description has been provided for this image
def band_pass(size, r1, r2):
    return ideal_low_pass(size, r1) - ideal_low_pass(size, r2)
img_s = iio.imread('space.png')
show_fft_fd_filter(img_s, band_pass(img_s.shape, 130, 120))
No description has been provided for this image
def band_reject(size, r1, r2):
    return (ideal_low_pass(size, r1) - ideal_low_pass(size, r2)) + 1
img_s = iio.imread('space.png')
show_fft_fd_filter(img_s, band_reject(img_s.shape, 120, 130))
No description has been provided for this image
 
 
 