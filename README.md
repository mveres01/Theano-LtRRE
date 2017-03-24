# Theano implementation of Learning to Remember Rare Events by Kaiser et al. (https://arxiv.org/abs/1703.03129)

## Key concepts
* Analogy to python dictionary
* Why use features of a network as dictionary keys
* Computing nearest neighbours for finding similar samples
* How and why the memory module handles rare cases
* Updating memory keys: via norm of cosine similarity, or overwriting old memory
* How triplet loss differs from classification loss
* GIF of memory module learning to remember events

At its core, the memory module resembles what is known as a *dictionary* in Python. A dictionary takes a key __k__, and maps it to a value __v__ by passing it through something known as a *hash function*. These hash functions take your data, and perform some operations to convert the key into a memory address, where it stores the value. In Python, this sort of structure can be built using the following syntax:

```python
key = 'cat'
value = 'animal'
dictionary = {key: value}
```
What happens here is that the key 'cat' gets mapped to a specific memory address, and the value at that address corresponds to 'animal'. 
Dictionaries are powerful data structures and allow very efficient queries and data lookups. So how do they fit in with Neural Networks? 

Neural Networks are function approximators that learn to take an input __X__, and map it to some output __Y__. One of the most popular use-cases for neural networks is in classification, which attempts to answer the question *what object is represented in this image?*. 

