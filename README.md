# ECA-Mixer

Pytorch implementation of paper: [Fast physic-informed mixer architecture for color Lensfree holographic reconstruction.](https://www.sciencedirect.com/science/article/abs/pii/S0143816623004268) 

## Citation
If you find this project useful, we would be grateful if you cite the **ECA-Mixer paper：**

Wang, Jiaqian, et al. "Fast physic-informed mixer architecture for color Lensfree holographic reconstruction." Optics and Lasers in Engineering 173 (2024): 107897.


## Abstract
Accurate color image reconstruction from multi-wavelength holograms is crucial in biomedical imaging applications. Current data-driven deep learning methods have made significant progress with promising reconstruction performance. Especially, untrained neural network methods have been developed to overcome data acquisition and the generalization issue. However, improving reconstruction quality requires more iterations, which can lead to longer convergence time. To address this challenge, we proposed an efficient complex-valued attention mixer (ECA-Mixer) architecture for fast and accurate physic-informed color holographic reconstruction. Our architecture consists of three core modules, an encoder, a nonlinear transformer and a decoder, each combining efficient attention mechanisms and Mixer layers for channel feature extraction and spatial information transformation. To preserve high-frequency information, we also introduced the use of 2D Haar wavelet and its inverse transform for encoding and decoding features. We then validate our method with extensive experiments on simulated and experimental samples and achieve state-of-art color reconstruction results in terms of computation time and image quality. Besides, we also demonstrate that our proposed lightweight network is capable of imaging large-size wide-field samples at a higher resolution in just a few minutes. The code is available and will be uploaded later at https://github.com/DeepPhysicVision/ECA-Mixer.git.

## Pipeline
![avatar]([https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41377-020-0302-3/MediaObjects/41377_2020_302_Fig1_HTML.png?as=webp](https://ars.els-cdn.com/content/image/1-s2.0-S0143816623004268-gr1_lrg.jpg))

## How to use
**Step 1: Configuring required packages**

python 3.8

pytorch 1.9.0

matplotlib 3.1.3

numpy 1.19

**Step 2: Run main.py after download and extract the ZIP file.**

## Results
![avatar]([https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41377-020-0302-3/MediaObjects/41377_2020_302_Fig5_HTML.png?as=webp](https://ars.els-cdn.com/content/image/1-s2.0-S0143816623004268-gr3_lrg.jpg)https://ars.els-cdn.com/content/image/1-s2.0-S0143816623004268-gr3_lrg.jpg)

## License
For academic and non-commercial use only.
