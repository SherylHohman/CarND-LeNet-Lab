
# coding: utf-8

# # LeNet Lab
# ![LeNet Architecture](lenet.png)
# Source: Yan LeCun

# ## Load Data
#
# Load the MNIST data, which comes pre-loaded with TensorFlow.
#
# You do not need to modify this section.

# In[1]:

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", reshape=False)
X_train, y_train           = mnist.train.images, mnist.train.labels
X_validation, y_validation = mnist.validation.images, mnist.validation.labels
X_test, y_test             = mnist.test.images, mnist.test.labels

assert(len(X_train) == len(y_train))
assert(len(X_validation) == len(y_validation))
assert(len(X_test) == len(y_test))

print()
print("Image Shape: {}".format(X_train[0].shape))
print()
print("Training Set:   {} samples".format(len(X_train)))
print("Validation Set: {} samples".format(len(X_validation)))
print("Test Set:       {} samples".format(len(X_test)))


# The MNIST data that TensorFlow pre-loads comes as 28x28x1 images.
#
# However, the LeNet architecture only accepts 32x32xC images, where C is the number of color channels.
#
# In order to reformat the MNIST data into a shape that LeNet will accept, we pad the data with two rows of zeros on the top and bottom, and two columns of zeros on the left and right (28+2+2 = 32).
#
# You do not need to modify this section.

# In[2]:

import numpy as np

# Pad images with 0s
X_train      = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_validation = np.pad(X_validation, ((0,0),(2,2),(2,2),(0,0)), 'constant')
X_test       = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')

print("Updated Image Shape: {}".format(X_train[0].shape))


# ## Visualize Data
#
# View a sample from the dataset.
#
# You do not need to modify this section.

# In[3]:

import random
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image, cmap="gray")
print(y_train[index])


# ## Preprocess Data
#
# Shuffle the training data.
#
# You do not need to modify this section.

# In[4]:

from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)


# ## Setup TensorFlow
# The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
#
# You do not need to modify this section.

# In[5]:

import tensorflow as tf

EPOCHS = 10
BATCH_SIZE = 128


# ## TODO: Implement LeNet-5
# Implement the [LeNet-5](http://yann.lecun.com/exdb/lenet/) neural network architecture.
#
# This is the only cell you need to edit.
# ### Input
# The LeNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since MNIST images are grayscale, C is 1 in this case.
#
# ### Architecture
# **Layer 1: Convolutional.** The output shape should be 28x28x6.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 14x14x6.
#
# **Layer 2: Convolutional.** The output shape should be 10x10x16.
#
# **Activation.** Your choice of activation function.
#
# **Pooling.** The output shape should be 5x5x16.
#
# **Flatten.** Flatten the output shape of the final pooling layer such that it's 1D instead of 3D. The easiest way to do is by using `tf.contrib.layers.flatten`, which is already imported for you.
#
# **Layer 3: Fully Connected.** This should have 120 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 4: Fully Connected.** This should have 84 outputs.
#
# **Activation.** Your choice of activation function.
#
# **Layer 5: Fully Connected (Logits).** This should have 10 outputs.
#
# ### Output
# Return the result of the 2nd fully connected layer.

# In[6]:

from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    padding = "VALID"
    def filter_size(in_size, out_size, stride):
        if padding == "VALID" :
            return (in_size + 1) - (out_size*stride)
        else:
            #print("PLEASE Change global padding to 'VALD'. \n     This Exercise requires changing dimensions")
            assert(padding == "VALID")
            return 0



    stride = 1  #2 stride of 2 isn't possible, given input and out dimensions we must match
    strides = [1, stride, stride, 1]

    # TODO: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.

    #print("data input : 32x32x1 =?=", x.get_shape())

    input_shape = x.get_shape()
    in_height = int(input_shape[1])
    in_width  = int(input_shape[2])
    in_depth  = int(input_shape[3])

    out_height = 28
    out_width  = 28
    out_depth  = 6

    # based on padding = "VALID"
    filter_h =  filter_size(in_height, out_height, stride)
    filter_w =  filter_size(in_width , out_width,  stride)
    shape    = [filter_h, filter_w, in_depth, out_depth]

    Filter_Weights = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
    Filter_Bias    = tf.Variable(tf.zeros([out_depth]))

    layer1 = tf.nn.conv2d(x, Filter_Weights, strides, padding) + Filter_Bias

    #print("\nlayer1 conv: 28x28x6 =?=", layer1.get_shape()[3])
    assert( [int(layer1.get_shape()[1]), int(layer1.get_shape()[2]), int(layer1.get_shape()[3]) ] == [28, 28, 6])

    # TODO: Activation.
    layer1 = tf.nn.relu(layer1)
    #print("layer1 RELU: 28x28x6 =?=", layer1.get_shape())
    assert( [int(layer1.get_shape()[1]), int(layer1.get_shape()[2]), int(layer1.get_shape()[3]) ] == [28, 28, 6])

    # TODO: Pooling. Input = 28x28x6. Output = 14x14x6.
    input_shape = layer1.get_shape()
    in_height = int(input_shape[1])
    in_width  = int(input_shape[2])
    in_depth  = int(input_shape[3])

    out_height = 14
    out_width  = 14
    out_depth  = in_depth

    # not certain what is the universal formula for height/width.
    # It works here, when stride == 1
    p_height = in_height - out_height + 1                            #15
    p_width  = in_width  - out_width  + 1                            #15
    # Below also works, when stride == 1
    #p_height = p_width = filter_size(in_height, out_height, stride) #15
    # Neither of above methods work when stride == 2
    # Below works when strides == 1 AND when strides == 2
    #ksize == strides   # works: [1,1,1,1], or [1,2,2,1]

    # 1st and last params are always 1, so as to not pool across channels, or batch field
    ksize = [1, p_height, p_width, 1]

    layer1 = tf.nn.max_pool(layer1, ksize, strides, padding)
    #print("layer1 pool: 14x14x6 =?=", layer1.get_shape())
    assert( [int(layer1.get_shape()[1]), int(layer1.get_shape()[2]), int(layer1.get_shape()[3]) ] == [14, 14, 6])

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    input_shape = layer1.get_shape()
    in_height = int(input_shape[1])
    in_width  = int(input_shape[2])
    in_depth  = int(input_shape[3])

    out_height = 10
    out_width  = 10
    out_depth  = 16

    filter_h =  filter_size(in_height, out_height, stride)
    filter_w=  filter_size(in_width , out_width,  stride)
    shape    = [filter_h, filter_w, in_depth, out_depth]

    Filter_Weights = tf.Variable(tf.truncated_normal(shape, mean=mu, stddev=sigma))
    Filter_Bias    = tf.Variable(tf.zeros([out_depth]))

    layer2 = tf.nn.conv2d(layer1, Filter_Weights, strides, padding) + Filter_Bias
    #print("\nlayer2 conv: 10x10x16 =?=", layer2.get_shape())
    assert( [int(layer2.get_shape()[1]), int(layer2.get_shape()[2]), int(layer2.get_shape()[3]) ] == [10, 10, 16])


    # TODO: Activation.
    layer2 = tf.nn.relu(layer2)
    #print("layer2 RELU: 10x10x16 =?=", layer1.get_shape())
    assert( [int(layer2.get_shape()[1]), int(layer2.get_shape()[2]), int(layer2.get_shape()[3]) ] == [10, 10, 16])

    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    input_shape = layer2.get_shape()
    in_height = int(input_shape[1])
    in_width  = int(input_shape[2])
    in_depth  = int(input_shape[3])

    out_height = 5
    out_width  = 5
    depth      = in_depth   # 16

    # not certain what is the universal formula for height/width. It works here, with stride==1, but not for stride==2
    p_height = in_height - out_height + 1
    p_width  = in_width  - out_width  + 1
    #p_height = p_width = filter_size(in_height, out_height, stride)  # works for stride==1
    #ksize = strides  # works for stride==1 AND stride==2
    ksize = [1, p_height, p_width, 1]

    layer2 = tf.nn.max_pool(layer2, ksize, strides, padding)
    #print("layer2 pool: 5x5x16 =?=", layer2.get_shape())
    assert( [int(layer2.get_shape()[1]), int(layer2.get_shape()[2]), int(layer2.get_shape()[3]) ] == [5, 5, 16])


    # TODO: Flatten. Input = 5x5x16. Output = 400.
    flat_23 = tf.contrib.layers.flatten(layer2)
    #print("\nflat_23: 400 =?=", flat_23.get_shape())
    assert( [int(flat_23.get_shape()[1]) ] == [400])

    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    in_height  = int(flat_23.get_shape()[1])
    out_height = 120

    weights = tf.Variable(tf.truncated_normal([in_height, out_height], mean=mu, stddev=sigma))
    biases  = tf.Variable(tf.zeros([out_height]))

    layer3 = tf.add(tf.matmul(flat_23,weights),biases)
    #print("\nlayer3: 120 =?=", layer3.get_shape())
    assert( [int(layer3.get_shape()[1]) ] == [120])

    # TODO: Activation.
    layer3 = tf.nn.relu(layer3)

    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    in_height = int(layer3.get_shape()[1])
    out_height = 84

    weights = tf.Variable(tf.truncated_normal([in_height, out_height], mean=mu, stddev=sigma))
    biases  = tf.Variable(tf.zeros([out_height]))

    layer4 = tf.add(tf.matmul(layer3,weights),biases)
    #print("layer4: 84 =?=", layer4.get_shape())
    assert( [int(layer4.get_shape()[1]) ] == [84])

    # TODO: Activation.
    layer4 = tf.nn.relu(layer4)

    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    in_height = int(layer4.get_shape()[1])
    out_height = 10

    weights = tf.Variable(tf.truncated_normal([in_height, out_height], mean=mu, stddev=sigma))
    biases  = tf.Variable(tf.zeros([out_height]))

    logits = tf.add(tf.matmul(layer4, weights),biases)
    #print("layer5: 10 =?=", logits.get_shape())
    assert( [int(logits.get_shape()[1]) ] == [10])


    return logits


# ## Features and Labels
# Train LeNet to classify [MNIST](http://yann.lecun.com/exdb/mnist/) data.
#
# `x` is a placeholder for a batch of input images.
# `y` is a placeholder for a batch of output labels.
#
# You do not need to modify this section.

# In[7]:

x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 10)


# ## Training Pipeline
# Create a training pipeline that uses the model to classify MNIST data.
#
# You do not need to modify this section.

# In[8]:

rate = 0.001

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)


# ## Model Evaluation
# Evaluate how well the loss and accuracy of the model for a given dataset.
#
# You do not need to modify this section.

# In[9]:

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# ## Train the Model
# Run the training data through the training pipeline to train the model.
#
# Before each epoch, shuffle the training set.
#
# After each epoch, measure the loss and accuracy of the validation set.
#
# Save the model after training.
#
# You do not need to modify this section.

# In[10]:

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")


# ## Evaluate the Model
# Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
#
# Be sure to only do this once!
#
# If you were to measure the performance of your trained model on the test set, then improve your model, and then measure the performance of your model on the test set again, that would invalidate your test results. You wouldn't get a true measure of how well your model would perform against real data.
#
# You do not need to modify this section.

# In[11]:

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))


# In[ ]:





