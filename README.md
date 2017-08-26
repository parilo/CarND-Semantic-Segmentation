# Road Semantic Segmentation with Full Convolutional Network on Kitti dataset

[Road surface segmentation](https://github.com/parilo/CarND-Semantic-Segmentation/blob/master/sample.png)

This is project 2 of Term 3 of Udacity Self-Driving Car Nanodegree. The goal of the project is to classify each pixel on the image if it is road or not. Such tasks are called semantic segmentation of the image.

### Full Convolutional Network (FCN)

In this project I use Full Convolutional Network. This network consist from convolutional part and deconvolutional part. For convolutional part I use pretrained VGG-16 with freezed weights. This network have several features:

* 1x1 convolutions. Fully connected layers replaced with mathematically same operation of 1x1 convolution that saves spatial information
* Transposed convolutions. Convolutions that makes layers bigger.
* Skip connections. Connections that passes outputs from some of layers from convolutional part into similar layers from deconvolutional parts. This is needed to save information about some small features.

Whole architecure looks like this:

[![FCN architecture](https://github.com/parilo/CarND-Semantic-Segmentation/blob/master/3-Figure3-1.png)](https://www.semanticscholar.org/paper/PCA-aided-Fully-Convolutional-Networks-for-Semanti-Tai-Ye/05d20ad124a8696f387e6c9632dec0b31251df64)

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

##### Run
Run the following command to run the project:
```
python main.py

type

python main.py -h

for details
```
