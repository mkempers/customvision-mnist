# Example of customvision.ai with MNIST dataset
Trying out customvision.ai with the MNIST dataset. I used Python 3.5 for this.

Custom Vision Service is a tool for building custom image classifiers. It makes it easy and fast to build, 
deploy, and improve an image classifier, see https://customvision.ai. Custom Vision Service is a tool for building 
custom image classifiers, and for making them better over time. The classifier is pre-trained for image recognition,
which means you need less data then with a self-trained machine learning classifier. Just thirty images per class 
should be enough. 

## MNIST database
The MNIST database consists of handwritten digits and has a training set of 60,000 examples, 
and a test set of 10,000 examples. It is a subset of a larger set available from NIST. 
The digits have been size-normalized and centered in a fixed-size image. See http://yann.lecun.com/exdb/mnist/.

## Running the example
Create an account on https://customvision.ai and copy the training key and the prediction key.

Replace constants TRAINING_KEY and PREDICTION_KEY in file cv_mnist.py with your values.

Install the necessary Python modules.
```
pip install -r requirements.txt
```

Import the MNIST data set. I have included a script for this.
```
python mnist-to-jpg.py
```

Run the training and prediction example
```
python mnist-to-jpg.py
```
This example:
1. Creates a new project.
2. Creates the necessary tags.
3. Imports the labels from the training images.
4. Uploads the training images to the Custom Vision service.
5. Trains custom vision to recognize characters
6. Runs a single test on the character '8'.

