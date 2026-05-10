Processamento de Imagens
Transformada de Fourier (Parte II)
Professora Leo Sampaio Ferraz Ribeiro

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
x = np.arange(-10, 10)
y = f(x, 2, 1, -4)
y
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[6], line 2
      1 x = np.arange(-10, 10)
----> 2 y = f(x, 2, 1, -4)
      3 y

NameError: name 'f' is not defined
plt.scatter(x, y)
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
Cell In[7], line 1
----> 1 plt.scatter(x, y)

NameError: name 'y' is not defined
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
Fundamentos para Transformada de Fourier
Vamos observar diferentes sons/frequências e a soma delas

freq = 440
x = np.arange(0, 1, 1/5000)
y = np.sin(2*np.pi*freq*x)
ipd.Audio(y, rate = 5000)
 
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

y = np.concatenate([make_sound(e4), make_sound(g4), make_sound(d4), make_sound(c4), make_sound(d4), make_sound(e4),
                   make_sound(g4), make_sound(d4)], axis=0)
ipd.Audio(y, rate = 44000)
plt.figure(figsize=(90, 10))
plt.plot(y[::500])
[<matplotlib.lines.Line2D at 0x17e9afed0>]
No description has been provided for this image
cm = make_sound(c4) + make_sound(e4) + make_sound(g4)
plt.plot(cm[:3000:10])
ipd.Audio(cm, rate = 44000)
No description has been provided for this image
Gostaríamos de criar um mecanismo para extração de frequências a partir de uma onda que soma diversas frequências

Nosso plano será multiplicar nosso sinal pela função bidimensional que define um círculo

def fun_(t):
    return np.sin(t*(1/10)*np.pi*2) + 1

def circulo(t, freq):
    return fun_(t)*np.cos(t*freq*np.pi*2), fun_(t)*np.sin(t*freq*np.pi*2)

def circulo_sem_f(t, freq):
    return np.cos(t*freq*np.pi*2), np.sin(t*freq*np.pi*2)

x, y = circulo(np.linspace(0, 50, 1000), freq=1/8)
plt.figure(figsize=(6, 6))
plt.scatter(x, y)
plt.scatter(*circulo_sem_f(np.linspace(0, 50, 1000), freq=1))
plt.xlim(-2, 2)
plt.ylim(-2, 2)
(-2.0, 2.0)
No description has been provided for this image
Controlamos então a frequência de uma volta no círculo e observamos o efeito em diferentes funções

signal = make_sound(4)
samples = np.arange(0, 1, 1/44000)

plt.figure(figsize=(10,25))
plt.tight_layout()
freqs = np.arange(0, 8, 0.2)
n_rows = len(freqs)//4
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal*np.sin(freq*samples*2*np.pi), signal*np.cos(freq*samples*2*np.pi)
    plt.subplot(n_rows, 4, i+1)
    plt.axis([-1, 1, -1, 1])
    plt.plot(x, y)
    plt.plot(x.mean(), y.mean(), 'ro')
    plt.title(f"Freq = {freq:.2}")
    centers_of_mass.append(x.sum())
plt.tight_layout()
No description has been provided for this image
Podemos observar algo interessante acontecendo quando as frequências do círculo se alinham com a frequência do sinal

# sinal que mistura 3 frequências
signal = make_sound(1) + make_sound(4) + make_sound(7)
samples = np.arange(0, 1, 1/44000)

plt.figure(figsize=(10,25))
plt.tight_layout()
freqs = np.arange(0, 8, 0.2)
n_rows = len(freqs)//4
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal*np.sin(freq*samples*2*np.pi), signal*np.cos(freq*samples*2*np.pi)
    plt.subplot(n_rows, 4, i+1)
    plt.axis([-1, 1, -1, 1])
    plt.plot(x, y)
    plt.plot(x.mean(), y.mean(), 'ro')
    plt.title(f"Freq = {freq:.2}")
    centers_of_mass.append(x.sum())
plt.tight_layout()
No description has been provided for this image
Podemos observar qual é o centro de massa no eixo x ao longo das diferentes frequências de círculos

Vamos ver os picos de centro de massa exatamente nas frequências que usamos para construir o sinal

plt.plot(freqs, centers_of_mass)
[<matplotlib.lines.Line2D at 0x30153fb10>]
No description has been provided for this image
Podemos aumentar a resolução das frequências observadas

E também observar a informação de centro de massa no eixo y (coseno)

freqs = np.arange(0, 600, 0.1)
signal = cm
centers_of_mass = []
for i, freq in enumerate(freqs):
    x, y = signal*np.cos(2*np.pi*freq*samples), signal*np.sin(2*np.pi*freq*samples)
    centers_of_mass.append([x.mean(), y.mean()])
plt.plot(freqs, centers_of_mass)
[<matplotlib.lines.Line2D at 0x3019b9350>,
 <matplotlib.lines.Line2D at 0x3019d1590>]
No description has been provided for this image
Podemos desfazer a transformação desenrolando o círculo

centers_of_mass = np.array(centers_of_mass)
signal_reconstructed = np.zeros_like(signal)
for i, freq in enumerate(freqs):
    signal_reconstructed += centers_of_mass[i, 0]*np.cos(2*np.pi*freq*samples) + centers_of_mass[i, 1]*np.sin(2*np.pi*freq*samples)

ipd.Audio(signal_reconstructed, rate=44000)
Definimos então formalmente a Transformada de Fourier

F
(
ω
)
=
N
∑
t
=
0
 
f
(
t
)
e
−
j
2
π
ω

f
(
t
)
=
N
∑
t
=
0
 
f
(
ω
)
e
j
2
π
t

def ft(signal, rate):
    n_samples = len(signal)
    freqs = np.arange(0, n_samples, 1)
    total_time = n_samples // rate
    t_samples = np.arange(0, total_time, total_time/n_samples)

    transform = np.zeros_like(signal, dtype=np.complex128)

    for freq in freqs[:rate//2]:
        for idx_t, t in enumerate(t_samples):
            transform[freq] += signal[idx_t]*np.exp(-1j*2*np.pi*freq*t)
    return transform
f = make_sound
lullaby = np.concatenate([f(e4, 2000), f(g4, 2000), f(d4, 2000), f(c4, 2000), f(d4, 2000), f(e4, 2000), f(g4, 2000), f(d4, 2000)])
f_lullaby = ft(lullaby, 2000)
plt.plot((abs(f_lullaby))[0:1000])
[<matplotlib.lines.Line2D at 0x302c9a8d0>]
No description has been provided for this image
Versão rápida (Fast Fourier Transform) permite que a gente use mais amostras do sinal

np.concatenate([f(e4, 44000), f(g4, 44000), f(d4, 44000), f(c4, 44000), f(d4, 44000), f(e4, 44000), f(g4, 44000), f(d4, 44000)])
plt.plot(abs(np.fft.fft(lullaby)))
[<matplotlib.lines.Line2D at 0x302d752d0>]
No description has been provided for this image
 