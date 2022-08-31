import numpy as np
from PIL import Image
import SituMeasure_step
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.reset_default_graph()

pixelsize = 1.85e-3
Z = 1
with tf.compat.v1.Session() as sess:
    dir = 'Phase'
    phase = Image.open(f'./Color/{dir}/P.bmp')
    TargetPhase = np.array(phase)
    TargetPhase = (TargetPhase - np.min(TargetPhase)) / (np.max(TargetPhase) - np.min(TargetPhase))  # scale to 0~1
    print(TargetPhase.shape)

    for i in range(3):
        #input = (1- tensor_amp * 0.5) * torch.exp(1j * np.pi * (1 - tensor_phase))
        input = TargetPhase[:,:,i]
        wave = [636e-6, 530e-6, 470e-6]
        wavelength = wave[i]
        out_measure = SituMeasure_step.AS(input, wavelength, pixelsize, Z)
        out_measure = sess.run(out_measure)
        diffraction = (out_measure - np.min(out_measure)) / (np.max(out_measure) - np.min(out_measure))
        diffraction = diffraction * 255
        diffraction = Image.fromarray(diffraction.astype('uint8')).convert('L')
        #diffraction = diffraction.astype('float32') * 255.0
        #cv2.imwrite(f'./Color/{dir}/{i}.bmp', diffraction)
        diffraction.save(f'./Color/{dir}/{i}.bmp')
