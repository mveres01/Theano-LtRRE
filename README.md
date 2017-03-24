# Theano implementation of Learning to Remember Rare Events by Kaiser et al. (https://arxiv.org/abs/1703.03129)

The memory module acts as a standalone component to a neural network, and serves as a medium for the network to store and retrieve external information. 

An analogy for this module is to think of it as a differentiable *dictionary*, where the keys of the dictionary are learned by the neural network. These keys represent high-level features of an input, and may be shared across many different inputs (i.e. so long as these features are similar). The values these keys are bound to represent the class labels of a given sample. 

## Dependencies:
* Numpy 1.12.1
* Theano 0.9 for running memory.py (See: http://deeplearning.net/software/theano/install.html)
* Lasagne 0.2 for running the example mnist.py (See: http://lasagne.readthedocs.io/en/latest/user/installation.html)

To run the example:

```python
python mnist.py
```

You should see something like:

```python
Starting training...
Epoch 1 of 2 took 56.404s
  training loss:                0.042020
  validation loss:              0.024000
  validation accuracy:          96.88 %
Epoch 2 of 2 took 55.051s
  training loss:                0.022224
  validation loss:              0.018059
  validation accuracy:          97.65 %
```
