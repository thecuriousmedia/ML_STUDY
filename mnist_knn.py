import numpy as  np
from collections import defaultdict
from tensorflow.examples.tutorials.mnist import input_data

## Distance Function used to calculate how two images are closed to each other
def euclidean_distance(img_a, img_b):
    '''Finds the distance between 2 images: img_a, img_b'''
    # element-wise computations are automatically handled by numpy
    return sum((img_a - img_b) ** 2)


def find_majority(labels):
    '''Finds the majority class/label out of the given labels'''
    # defaultdict(type) is to automatically add new keys without throwing error.
    counter = defaultdict(int)
    for label in labels:
        counter[label] += 1

    # Finding the majority class.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
           return key

mnist = input_data.read_data_sets('/home/karan/Downloads/KARAN/NN/tensorflow/mnist_data')
train_images = np.asarray(mnist.train.images[:5000])
train_labels = np.asarray(mnist.train.labels[:5000])
test_images = np.asarray(mnist.test.images[:1000])
test_labels = np.asarray(mnist.test.labels[:1000])

def predict(k, train_images, train_labels, test_images):
    '''
    Predicts the new data-point's category/label by 
    looking at all other training labels
    '''
    # distances contains tuples of (distance, label)
    distances = [(euclidean_distance(test_image, image), label)
                    for (image, label) in zip(train_images, train_labels)]
    print (distances)
    # sort the distances list by distances
  #  by_distances = sorted(distances, key=lambda(distance, _): distance)
   # by_distances = sorted(distances, key=lambda distance:)
    by_distances = sorted(distances, key=lambda distance : distance)
    # extract only k closest labels
    k_labels = [label for (_, label) in by_distances[:k]]
    # return the majority voted label
    return find_majority(k_labels)

# Predicting and printing the accuracy
i = 0
total_correct = 0
for test_image in test_images:
    pred = predict(10, train_images, train_labels, test_image)
    if pred == test_labels[i]:
        total_correct += 1
    acc = (total_correct / (i+1)) * 100
    print('test image['+str(i)+']', '\tpred:', pred, '\torig:', test_labels[i], '\tacc:', str(round(acc, 2))+'%')
    i += 1

