# Problema del coctel
 LABORATORIO - 3 PROCESAMIENTO DIGITAL DE SEÑALES


## Requisitos
- Python 3.9
- Bibliotecas necesarias:
  - numpy
  - matplotlib
  - scipy.io
  - sounddevice
  - FastICA
 
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd
from sklearn.decomposition import FastICA
```

## Introducción
El problema de la "fiesta de cóctel" es un fenómeno estudiado en procesamiento de señales que hace referencia a la capacidad de un sistema para concentrarse en una sola fuente sonora mientras filtra las demás en un entorno con múltiples emisores de sonido. Este problema es de gran relevancia en aplicaciones de mejora de la inteligibilidad del habla, reconocimiento automático de voz y cancelación de interferencias acústicas en entornos ruidosos.

En este laboratorio, mediante un arreglo de  tres micrófonos estrategicamente distribuidos en el espacio se capturarán tres señales de voz. Estas señales adquiridas se analizarán de manera espectral utilizando transformada de Fourier discreta (DFT) o la transformada rápida de Fourier (FFT), describiendo la información que se puede obtener con cada una de ellas. Además, se implementarán técnicas previamente investigadas de separación de señales como el Análisis de Componentes Independientes (ICA) o el Beamforming con el objetivo de aislar la señal de interés a partir de las señales capturadas por los micrófonos. 

A lo largo de este experimento,  se evaluará el impacto de la disposición espacial de los micrófonos en la efectividad de la separación de señales, así como analizar métricas cuantitativas, como la relación señal-ruido (SNR), para medir el desempeño de los métodos aplicados.

---

## Importación de audios y ruidos 
```python
fs1, micro1 = wavfile.read("audioana.wav")
fsr1, ruido1 = wavfile.read("ruidoana.wav")

fs2, micro2 = wavfile.read("Audiosantiago.wav")
fsr2, ruido2 = wavfile.read("ruidosantiago.wav")

fs3, micro3 = wavfile.read("audiosamuel.wav")
fsr3, ruido3 = wavfile.read("ruidosamuel.wav")
```
Este código lee archivos de audio en formato WAV utilizando wavfile.read de scipy.io.wavfile. Y se cargan dos archivos: uno con la señal de las voces (micro) y otro con el ruido ambiental (ruido), junto con sus respectivas frecuencias de muestreo (fs y fsr).

### 1. Calculo Del SNR
```python
def SNR(m,r):
  potencia_señal = np.mean(m**2)
  potencia_ruido = np.mean(r**2)
  
  snr = 10 * np.log10(potencia_señal/potencia_ruido)
  return snr

print('SNR micrófono 1:',round(SNR(micro1,ruido1),3),'dB')
print('SNR micrófono 2:', round(SNR(micro2,ruido2),3),'dB')
print('SNR micrófono 2:', round(SNR(micro3,ruido3),3),'dB')
```
Este código calcula la relación señal-ruido (SNR) en decibeles (dB) para tres micrófonos. La función SNR(m, r) recibe dos arreglos NumPy: m, que representa la señal del micrófono, y r, que representa el ruido. Calcula la potencia de la señal y el ruido como la media de sus cuadrados y luego obtiene el SNR usando la fórmula.

$$
SNR = 10 \cdot \log_{10} \left( \frac{P_{\text{señal}}}{P_{\text{ruido}}} \right)
$$

Finalmente, se imprimen los valores de SNR redondeados a tres decimales para cada micrófono.

$$
SNR_{mic1} = 8.148 \text{ dB}
$$

$$
SNR_{mic2} = 14.267 \text{ dB}
$$

$$
SNR_{mic3} = 8.39 \text{ dB}
$$


### 2. Reproducción de los audios
```python
def reproducir_audio(audio, fs):
    sd.play(audio, fs)
    sd.wait()

reproducir_audio(micro1, fs1)
reproducir_audio(micro2, fs2)
reproducir_audio(micro3, fs3)
```

El código define una función llamada reproducir_audio que toma un archivo de audio y su frecuencia de muestreo (fs) como parámetros, reproduciéndolo con la biblioteca sounddevice. Luego, se llama a esta función tres veces para reproducir los audios micro1, micro2 y micro3 con sus respectivas frecuencias de muestreo (fs1, fs2, fs3). La función sd.wait() asegura que cada audio termine de reproducirse antes de iniciar el siguiente.

---
## Análisis Temporal 

### 1. Calculo de varianza y media
```python
    print('Media micrófono 1:', round(np.mean(micro1),3))
    print('Varianza micrófono 1:', round(np.var(micro1,ddof=1),3))
   
    print('Media micrófono 2:', round(np.mean(micro2),4))
    print('Varianza micrófono 2:', round(np.var(micro2,ddof=1),3))
   
    print('Media micrófono 3:', round(np.mean(micro3),3))
    print('Varianza micrófono 3:', round(np.var(micro3,ddof=1),3))
```
El código calcula y muestra la media y la varianza de los datos de los tres micrófonos (micro1, micro2 y micro3). Usa np.mean() para obtener la media y np.var(..., ddof=1) para la varianza muestral. Los resultados se redondean a tres o cuatro decimales antes de imprimirse.

$$
\mu_{mic1} = -0.025
$$

$$
\sigma^2_{mic1} = 16,529,446
$$

$$
\mu_{mic2} = -0.001
$$

$$
\sigma^2_{mic2} = 426,058
$$

$$
\mu_{mic3} = -0.003
$$

$$
\sigma^2_{mic3} = 6,089,800
$$



### 2. Grafica de los audios
```python
    t1 = np.linspace(0, len(micro1) / fs1, len(micro1))
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(t1,micro1,color='y')
    plt.title("Señal microfono 1")  
    plt.xlabel("tiempo [s]") 
    plt.ylabel("Amplitud digital (int16)") 
    plt.grid()
    
    t2 = np.linspace(0, len(micro2) / fs2, len(micro2))
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(t2,micro2,color='b')
    plt.title("Señal microfono 2")  
    plt.xlabel("tiempo [s]") 
    plt.ylabel("Amplitud digital (int16)") 
    plt.grid()
    
    t3 = np.linspace(0, len(micro3) / fs2, len(micro3))
    fig = plt.figure(figsize=(10, 5)) 
    plt.plot(t3,micro3,color='r')
    plt.title("Señal microfono 3")  
    plt.xlabel("tiempo [s]") 
    plt.ylabel("Amplitud digital (int16)") 
    plt.grid()
```
<p align="center">
    <img src="https://github.com/user-attachments/assets/e01e61e9-b2b7-4b56-895d-7daac9d08063" alt="image" width="450">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/52bfce35-f418-456f-b8c7-991cb7bd6e46" alt="image" width="450">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/8bb660e4-c995-426c-b580-9b63b38cfe7a" alt="image" width="450">
</p>


El código genera y muestra las gráficas de las señales de audio captadas por los tres micrófonos. Para cada micrófono, se crea un vector de tiempo (t1, t2, t3) usando np.linspace(0, duración, número de muestras), donde la duración se obtiene dividiendo la cantidad de muestras (len(microX)) por la frecuencia de muestreo (fsX). Esto permite representar la señal en el dominio del tiempo. Luego, plt.plot() grafica cada señal

---
## Análisis Espectral

### 1. Transformada rapida de fourier
```python
frecuencias = np.fft.fftfreq(len(micro1), 1/fs1)
spectro = np.abs(np.fft.fft(micro1)) / len(micro1)
```

Este código realiza un análisis de frecuencia de una señal de audio utilizando la Transformada Rápida de Fourier (FFT). Primero, con np.fft.fftfreq(len(micro1), 1/fs1), se generan las frecuencias correspondientes a cada componente de la FFT, donde len(micro1) es el número de muestras de la señal y 1/fs1 es el periodo de muestreo, determinado por la frecuencia de muestreo fs1. Luego, con np.fft.fft(micro1), se calcula la FFT de la señal micro1, transformándola del dominio del tiempo al dominio de la frecuencia. Posteriormente, se obtiene la magnitud del espectro con np.abs(...), descartando la información de fase, y finalmente, se normaliza dividiendo entre len(micro1). Como resultado, se obtiene el espectro de frecuencias de la señal, lo que permite analizar su contenido en términos de amplitud y frecuencia.

### 2. Graficas
```python
fig = plt.figure(figsize=(8, 4)) 
plt.plot(frecuencias[:len(micro1)//2], spectro[:len(micro1)//2],color='y')
plt.title("Transformada de fourier de la señal 1 (espectro)")  
plt.xlabel("Frecuencias [Hz]") 
plt.ylabel("Amplitud Normalizada")
plt.grid()
plt.show()
```
Este código genera y muestra el espectro de frecuencias de la señal micro1. Se grafica la mitad del espectro positivo con plt.plot(...), usando la frecuencia en el eje X y la amplitud normalizada en el eje Y. Cabe mencionar que es solo de la transformada de un señal, pero en el codigo se calculan la transformada de cada una de las tres señales grabadas

<p align="center">
    <img src="https://github.com/user-attachments/assets/2d6e6627-0a52-4e5a-80dc-5114d79baa31" alt="image" width="450">
</p>

<p align="center">
    <img src="https://github.com/user-attachments/assets/372acc0f-31f2-4673-8322-478f1fa41777" alt="image" width="450">
</p>




---
## Separación de Voces

----
## Conclusión



----
## Bibliografias

----
## Autores 
- Samuel Peña
- Ana Abril
- Santiago Mora
