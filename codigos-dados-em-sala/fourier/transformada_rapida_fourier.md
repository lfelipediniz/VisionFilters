Processamento de Imagens
Transformada de Fourier (Parte IV)
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

def show_fft(ogimg, show_phase=False):
    plt.figure(figsize=(15, 30))
    f_img = np.log2(np.fft.fftshift(np.fft.fft2(ogimg))+0.000001).real
    plt.subplot(1, 2 if not show_phase else 3, 1)
    plt.imshow(ogimg, cmap='gray')
    plt.subplot(1, 2 if not show_phase else 3, 2)
    plt.imshow(f_img, cmap='gray')
    if show_phase:
        f_img = np.fft.fft2(ogimg)
        plt.subplot(1, 3, 3)
        plt.imshow(np.arctan2(f_img.imag, f_img.real), cmap='gray')
Transformada Rápida de Fourier 1D
def make_sound(freq, rate=44000):
    x = np.arange(0, 1, 1/rate)
    y = np.sin(2*np.pi*freq*x)
    return y

c4 = 261.63
d4 = 293.66
e4 = 329.63
f4 = 349.23
g4 = 392.00
a4 = 440.00
b4 = 493.88

rate = 2**14
y = np.concatenate([make_sound(e4, rate=rate), make_sound(g4, rate=rate), make_sound(d4, rate=rate), 
                    make_sound(c4, rate=rate), make_sound(d4, rate=rate), make_sound(e4, rate=rate),
                   make_sound(g4, rate=rate), make_sound(d4, rate=rate)], axis=0)
ipd.Audio(y, rate = 44000)
def fft(y):
    N = len(y)

    if N == 1:
        return y
    
    freqs = np.arange(N//2)

    F_even = fft(y[::2])
    F_odd = fft(y[1::2])
    W = np.exp(-1j*2*np.pi*freqs/N)

    return np.concatenate([F_even + F_odd*W, F_even - F_odd*W])
plt.plot(np.abs(fft(y))[0:5000])
[<matplotlib.lines.Line2D at 0x10e1f1d90>]
No description has been provided for this image
Transformada Rápida de Fourier 2D
def fft2d(img):
    F_h = np.zeros_like(img, dtype=np.complex128)
    F = np.zeros_like(img, dtype=np.complex128)
    H, W = img.shape
    for i in range(H):
        F_h[i, :] = fft(img[i, :])
    for j in range(W):
        F[:, j] = fft(F_h[:, j])
    return F
img = iio.imread('pompom_brownie.jpeg')[0:256, 0:256, :]
img = luminosity(img)
img = np.pad(img, ((256-img.shape[0], 0), (0, 0)))
plt.imshow(np.log2(np.fft.fftshift(np.abs(fft2d(img))+0.0001)), cmap='gray')
<matplotlib.image.AxesImage at 0x10f2cfdd0>
No description has been provided for this image
show_fft(img)
No description has been provided for this image
 