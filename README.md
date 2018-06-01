# cnn-autoencoder-tf
CNN and Autoencoder on MNIST

I added more parameters (lr: learning rate, mm: moemntum, bsz: batch size) and task for cross validation

### TASK 1 - CNN

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

### TASK 2 - Autoencoder

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
