"""
Demonstration of incorporating a memory module into a standard classification
network.

Base code for getting mnist up and running courtest of the Lasagne mnist
examples: https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
"""

import os
import sys
import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from memory import MemoryModule


def load_mnist():

    # We first define a download function, supporting both Python 2 and 3.
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve

    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print "Downloading %s" % filename
        urlretrieve(source + filename, filename)

    # We then define functions for loading MNIST images and labels.
    # For convenience, they also download the requested files if needed.
    import gzip

    def load_mnist_images(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the inputs in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        if not os.path.exists(filename):
            download(filename)
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_network(input_var, image_size=28, output_dim=10):

    nonlin = lasagne.nonlinearities.rectify
    W_init = lasagne.init.GlorotUniform()
    b_init = lasagne.init.Constant(0.)

    input_shape = (None, 1, image_size, image_size)

    network = nn.InputLayer(input_shape, input_var)

    network = nn.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                             nonlinearity=nonlin, W=W_init, b=b_init)
    network = nn.Conv2DLayer(network, num_filters=64, filter_size=(3, 3),
                             nonlinearity=nonlin, W=W_init, b=b_init)
    network = nn.MaxPool2DLayer(network, pool_size=(2, 2))

    network = nn.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.Conv2DLayer(network, num_filters=128, filter_size=(3, 3),
                             W=W_init, b=b_init, nonlinearity=nonlin)
    network = nn.MaxPool2DLayer(network, pool_size=(2, 2))

    network = nn.dropout(network, p=0.5)
    network = nn.DenseLayer(network, num_units=256, W=W_init, b=b_init,
                            nonlinearity=nonlin)

    network = nn.dropout(network, p=0.5)
    network = nn.DenseLayer(network, num_units=output_dim, W=W_init, b=b_init,
                            nonlinearity=None)

    return network


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def main():

    lrate = 1e-3
    batch_size = 32
    key_size = 256
    mem_size = 50*50
    k_nbrs = 128
    num_epochs = 100
    input_var = T.tensor4('x')
    target_var = T.ivector('y')

    print 'Loading data and creating train/test splits... '
    X_train, y_train, X_val, y_val, X_test, y_test = load_mnist()

    # Build our 'encoding' network
    network = build_network(input_var, image_size=X_train.shape[-1], 
                            output_dim=key_size)
    network_embedding = nn.get_output(network, deterministic=False)

    # Initialize the module and compile graphs for training. 
    # Note that this is where the difference between traditional neural network
    # classifiers comes in. Rather then computing a logistic regression, we use
    # the output of the memory module and triplet loss.
    MM = MemoryModule(mem_size, key_size, k_nbrs)
    mem_loss, mem_updates = MM.build_loss_and_updates(network_embedding, target_var)
    mem_loss = mem_loss.mean()

    # Use the Adam optimizer for training.
    params = nn.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(mem_loss, params, lrate, beta1=0.9)

    # Whenever we update the network parameters, we'll also update the memory
    # within the memory module
    updates.update(mem_updates)

    train_fn = theano.function([input_var, target_var], mem_loss, updates=updates)


    
    # For validation, we'll follow a deterministic mapping
    determ_embedding = nn.get_output(network, deterministic=True)
    mem_pred, _ = MM.query(determ_embedding)

    test_acc = T.mean(T.eq(mem_pred, target_var), dtype=theano.config.floatX)
    valid_fn = theano.function([input_var, target_var], [mem_loss, test_acc])


    # Finally, launch the training loop.
    print 'Starting training...'

    # We iterate over epochs:
    for epoch in range(num_epochs):

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch

            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch

            err, acc = valid_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print "Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time)
        print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
        print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
        print "  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100)


    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = valid_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print "Final results:"
    print "  test loss:\t\t\t{:.6f}".format(test_err / test_batches)
    print "  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100)


if __name__ == '__main__':
    main()
