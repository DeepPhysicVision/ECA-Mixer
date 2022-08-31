import tensorflow as tf
import numpy as np
import math
import SituMyfftshift

def AS(inpt,lamb,pixelsize,Z):
    M = int(inpt.shape[-1])
    N = int(inpt.shape[-2])

    image = tf.cast(inpt,dtype = tf.complex128)
    image = 1j*image
    U_in  = tf.exp(image)

    U_out=SituMyfftshift.ifftshift(tf.compat.v1.fft2d(SituMyfftshift.fftshift(U_in)))

    L = M * pixelsize
    fx=1/L
    x = np.linspace(-M/2,M/2-1,M) 
    fx = fx*x

    L1 = N * pixelsize
    fy=1/L1
    y = np.linspace(-N/2,N/2-1,N)
    fy = fy*y

    [Fx,Fy]=np.meshgrid(fx,fy)
    
    k = 2*math.pi/lamb 
    H = tf.sqrt(1-lamb*lamb*(Fx*Fx+Fy*Fy))
    temp = k*Z*H
    temp = tf.cast(temp,dtype = tf.complex64)
    
    H = tf.exp(1j*temp)
    U_out = U_out*H
        
    U_out = SituMyfftshift.ifftshift(tf.compat.v1.ifft2d(SituMyfftshift.fftshift(U_out)))
    I1 = tf.abs(U_out)*tf.abs(U_out)
    I1 = I1/tf.reduce_max(tf.reduce_max(I1))
    
    return I1


def AS_complex(phase, lamb, pixelsize, Z):
    M = int(phase.shape[-1])
    N = int(phase.shape[-2])

    image = tf.cast(phase, dtype=tf.complex128)
    image = 1j * image
    U_in = tf.exp(image)

    U_out = SituMyfftshift.ifftshift(tf.compat.v1.fft2d(SituMyfftshift.fftshift(U_in)))

    L = M * pixelsize
    fx = 1 / L
    x = np.linspace(-M / 2, M / 2 - 1, M)
    fx = fx * x

    L1 = N * pixelsize
    fy = 1 / L1
    y = np.linspace(-N / 2, N / 2 - 1, N)
    fy = fy * y

    [Fx, Fy] = np.meshgrid(fx, fy)

    k = 2 * math.pi / lamb
    H = tf.sqrt(1 - lamb * lamb * (Fx * Fx + Fy * Fy))
    temp = k * Z * H
    temp = tf.cast(temp, dtype=tf.complex64)

    H = tf.exp(1j * temp)
    U_out = U_out * H

    U_out = SituMyfftshift.ifftshift(tf.compat.v1.ifft2d(SituMyfftshift.fftshift(U_out)))
    I1 = tf.abs(U_out) * tf.abs(U_out)
    I1 = I1 / tf.reduce_max(tf.reduce_max(I1))

    return I1

