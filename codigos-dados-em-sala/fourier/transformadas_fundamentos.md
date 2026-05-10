Processamento de Imagens
Transformada de Fourier
Professora Leo Sampaio Ferraz Ribeiro

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
Representações de Funções
def f(x, a, b, c):
    return c + b*x + a*(x**2)
Amostragem da função

x = np.arange(-10, 10)
y = f(x, 2, 1, -4)
y
array([186, 149, 116,  87,  62,  41,  24,  11,   2,  -3,  -4,  -1,   6,
        17,  32,  51,  74, 101, 132, 167])
plt.scatter(x, y)
<matplotlib.collections.PathCollection at 0x109f36e10>
No description has been provided for this image
Construindo o sistema linear

A = np.zeros((20, 3))
A
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])
A[:, 0] = x**2
A[:, 1] = x**1
A[:, 2] = x**0
A
array([[100., -10.,   1.],
       [ 81.,  -9.,   1.],
       [ 64.,  -8.,   1.],
       [ 49.,  -7.,   1.],
       [ 36.,  -6.,   1.],
       [ 25.,  -5.,   1.],
       [ 16.,  -4.,   1.],
       [  9.,  -3.,   1.],
       [  4.,  -2.,   1.],
       [  1.,  -1.,   1.],
       [  0.,   0.,   1.],
       [  1.,   1.,   1.],
       [  4.,   2.,   1.],
       [  9.,   3.,   1.],
       [ 16.,   4.,   1.],
       [ 25.,   5.,   1.],
       [ 36.,   6.,   1.],
       [ 49.,   7.,   1.],
       [ 64.,   8.,   1.],
       [ 81.,   9.,   1.]])
y.transpose()
array([186, 149, 116,  87,  62,  41,  24,  11,   2,  -3,  -4,  -1,   6,
        17,  32,  51,  74, 101, 132, 167])
Resolvendo o sistema linear

import numpy.linalg as linalg

c_1 = linalg.inv(A.transpose()@A)
c_1
array([[ 5.69605833e-05,  5.69605833e-05, -1.87969925e-03],
       [ 5.69605833e-05,  1.56071998e-03, -1.12781955e-03],
       [-1.87969925e-03, -1.12781955e-03,  1.12406015e-01]])
C = c_1@(A.transpose()@y)
C
array([ 2.,  1., -4.])
Sinusoides e a Exponencial Complexa
Desejamos funções com a propriedade 
f
(
t
)
=
f
(
t
+
k
T
)

Podemos observar que senos e cosenos tem essa propriedade

x = np.arange(-5, 5, 0.2)
np.sin(x)
plt.scatter(x, np.sin(x+2*np.pi))
plt.scatter(x, np.cos(x+2*np.pi))
<matplotlib.collections.PathCollection at 0x10acca310>
No description has been provided for this image
O objeto periódico mais simples: círculo

circle = np.concatenate([np.sin(x)[:, None], np.cos(x)[:, None]], axis=1)
plt.figure(figsize=(4, 4))
plt.scatter(circle[:, 0], circle[:, 1])
<matplotlib.collections.PathCollection at 0x10ad132d0>
No description has been provided for this image
a = np.sin(x) + np.cos(x) + 0.2*np.sin(2*x) + 0.2*np.cos(2*x) + 0.05*np.sin(10*x) + 0.05*np.cos(10*x)
plt.scatter(x, a)
<matplotlib.collections.PathCollection at 0x10adb1590>
No description has been provided for this image
Números complexos

b = 1 + 3j
b.imag
3.0
Fórmula de Euler

# comparando resultado da exponencial com sen, cos
np.exp(0.1j) 
np.complex128(0.9950041652780258+0.09983341664682815j)
np.sin(0.1)

np.sin(0.1)
np.cos(0.1)
np.float64(0.9950041652780258)
 
Fundamentos para Transformada de Fourier
Vamos observar diferentes sons/frequências e a soma delas

freq = 440
x = np.arange(0, 1, 1/44000)
y = np.sin(2*np.pi*freq*x)
ipd.Audio(y, rate = 44000)
plt.figure(figsize=(500, 50))
plt.plot(x, y)
[<matplotlib.lines.Line2D at 0x10a998050>]
No description has been provided for this image
def make_sound(freq):
    x = np.arange(0, 1, 1/44000)
    y = np.sin(2*np.pi*freq*x)
    return y

c4 = 261.63
d4 = 293.66
e4 = 329.63
f4 = 349.23
g4 = 392.00
a4 = 440.00
b4 = 493.88

y = np.concatenate([make_sound(e4), make_sound(g4), make_sound(d4), make_sound(c4), make_sound(d4), make_sound(e4),
                   make_sound(g4), make_sound(d4)], axis=0)
ipd.Audio(y, rate = 44000)
y = make_sound(c4) + make_sound(e4) + make_sound(g4)
plt.plot(x, y)
ipd.Audio(y, rate = 44000)
No description has been provided for this image
Gostaríamos de criar um mecanismo para extração de frequências a partir de uma onda que soma diversas frequências