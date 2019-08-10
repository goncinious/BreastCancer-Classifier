# Automating breast cancer diagnosis in histopathology


- Build ML-based classifier to reduce inspection time of pathologists.
- Dataset was provided by [ICIAR2018 Challenge](https://iciar2018-challenge.grand-challenge.org).
- Data comprised by 400 images [1535x2048x3] of 4 classes (100 each): Benign, InSitu, Invasive, Normal.


Two approaches developed:
1. Machine learning: Extract Fisher Vector representation from each image and train a SVM classifier.
2. Deep learning: Learn representation automatically using a Convolutional Neural Networks (CNNs).

## Jupyter notebooks:

### 1) Reading and visualising images
[1-Data_Visualisation.ipynb](/notebooks/1-Data_Visualisation.ipynb)

### 2) Classification: FisherVector+SVM (80% test acc.)
[2-FisherVector_SVM.ipynb](/notebooks/2-FisherVector_SVM.ipynb)

### 3) Classification: CNN (93% test acc.)
[3-CNN1.ipynb](/notebooks/3-ConvNet1.ipynb)


![alt text](/src/utils/class_examples.png)

