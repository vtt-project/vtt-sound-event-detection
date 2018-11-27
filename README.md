# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)

# Introduction
This repository contains source code in Tensorflow of a simple sound event detection (multi-label classification) model using CNN+RNN. 
# Requirements 
* Python 2.7
* Tensorflow 1.9 
* Librosa 0.6
# Method 
## Preprocessing 
Each fixed-length frame of mono audio is preprocessed by short-time fourier transform (STFT) to produce spectrogram. This spectorgram then is fed to the CNN part to extract deep feature and reduce the size of the FFT. 
Due to the 2D-convolution in the CNN, a new channel dimension must be added to the spectrogram. Shape of the input: `batch_size x n_fft_frame x (N_FFT/2 + 1) x 1`
For example, with the parameter used in `vggish_rnn.py`, the shape of the extracted feature is `32 x 44 x 1025 x 1` 
## Model architecture 
### CNN as deep feature extractor
Instead of using mel frequency cesptral coefficients as the final feature, a CNN is used to extract deep features from the spectrogram. This CNN contains 7 3x3 convolution layers, average pooling, batch normalization layers. For more details, please refer to `model.py` 
### RNN as sequence classifier
A RNN takes the extracted features as the input then product the final output label. The activation of the output layer is `sigmoid`. 
# Result 
The above model is trained with audio data from 7 first episodes of Friends series then evaluated in the next 3 episodes. This data set is of high imbalance (10 classes, 3 dominant classes account more than 90% of the data). 
All audio of train and validation episodes are split to 1s clips with multiple labels. 
Main metric to measure the performance is `f1_score`. 

|           | background music | speaking | background laughing |
|-----------|------------------|----------|---------------------|
| precision | 0.67             | 0.93     | 0.51                |
| recall    | 0.70             | 0.85     | 0.86                |
| f1        | 0.68             | 0.88     | 0.63                |
