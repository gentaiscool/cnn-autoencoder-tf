# Convolutional Neural Network and Constractive Autoencoder on EMNIST

In this project, we are going to evaluate the performance of convolutional neural network (CNN) and constractive autoencoder (CAE) models by conducting empirical study on simple image data (EMNIST dataset) [1]. This dataset consists of 28x28 images of handwritten characters that belong to 47 classes.

[1] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre van Schaik. EMNIST: an
extension of MNIST to handwritten letters. arXiv preprint arXiv:1702.05373, 2017.

### Run the code
#### Package required
- Python 3.5 (or later)
- Tensorflow (https://www.tensorflow.org/)

#### Parameters
- lr: initial learning rate
- mm: momentum
- bsz: batch size

### CNN

#### Run the code
Train CNN
```
python --task="train_cnn" --lr=0.1 --mm=0.2 --bsz=32
```

Cross Validation CNN
```
python --task="cross_valid_cnn"
```

Test CNN
```
python --task="test_cnn" --lr=0.1 --mm=0.2 --bsz=32
```

### Autoencoder

<img src="fig/sample.png" height=350/>

#### Run the code
Train AE
```
python --task="train_ae" --lr=0.1 --mm=0.2 --bsz=32
```

Cross Validation AE
```
python --task="cross_valid_ae"
```

Test AE
```
python --task="evaluate_ae" --lr=0.1 --mm=0.2 --bsz=32
```

### Note

COMP5212 - Machine Learning Programming Assignment 2 in HKUST
