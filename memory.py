import os
import sys
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

def norm(X, eps=1e-8, keepdims=True):
    return X / (np.sqrt(np.sum(X**2, axis=1, keepdims=keepdims)) + eps)

def tensor_norm(X, eps=1e-8, keepdims=True):
    return X / (T.sqrt(T.sum(X**2, axis=1, keepdims=keepdims)) + eps)

def tensor_choose_k(boolean_mask, rng, k=1, random=False):
    """Selects an element from each row of boolean_mask.
  
    Parameters
    ----------
    boolean_mask: 2d Boolean matrix, where rows are samples and columns are
        possible choices.
    rng: An instance of theano.sandbox.rng_mrg.MRG_RandomStreams
    k (int): Number of samples we choose
    random: Boolean flag indicating whether to select 'k' indices in order 
        (random=False), or whether to randomly choose 'k' indices
        (randome=True).
    """
    mask = boolean_mask
    if mask.ndim > 2:
        raise Exception('Input tensor must be either 1d or 2d.')
    elif mask.ndim == 1:
        mask = mask.dimshuffle('x', 0)

    assert T.lt(k, mask.shape[1]), 'k must be leq number of possible choices'

    if random is True:
        noise = rng.uniform(mask.shape, low=0, high=mask.shape[1])
    else:
        noise = (1 + T.arange(mask.shape[1]))[::-1] # Descending order
        noise = noise.dimshuffle('x', 0)
   
    if k == 1:
        return T.argmax(mask*noise, axis=1)
    return T.argsort(mask*noise, axis=1)[:, ::-1][:, :k]

class MemoryModule(object):
    """Implements a memory module using theano.

    Note: It is important that every key is initialized differently.
          If all keys have the same unit norm, the module will fail to update
          properly as each query will produce the same nearest neighbours.
    Note: When training with neural networks, make sure that the size of memory
          is larger then the batch size, or this module will throw an error when
          trying find too many indexs' to memory.
    """

    def __init__(self, mem_size, key_size, k_nbrs, alpha=0.1, t=40, seed=1234,
                 C=3, K=None, V=None, A=None):

        assert k_nbrs <= mem_size, 'k_nbrs must be less then (or equal) mem size'

        self.mem_size = mem_size
        self.key_size = key_size
        self.k_nbrs = k_nbrs
        self.alpha = alpha
        self.t = t # Softmax temperature
        self.C = C # Threshold for sampling mamx ages during update
        self.rng = RandomStreams(seed=seed)

        if K is None:
            K = norm(np.random.randn(mem_size, key_size))
            K = K.astype(theano.config.floatX)
        else:
            if not np.allclose(np.sum(K**2, axis=1), np.ones(K.shape[0])):
                raise ValueError('The rows of K must be unit norm')
        self.K = theano.shared(K, name='keys')

        if V is None:
            V = np.ones(mem_size, dtype='int32')*-1
        self.V = theano.shared(V, name='values')

        if A is None:
            A = np.zeros(mem_size, dtype=theano.config.floatX)
        self.A = theano.shared(A, name='ages')

    def _format_query(self, query):
        """Convenience function for formatting query vector / matrix."""

        assert T.le(query.ndim, 2), 'Query must either be 1d (single '\
                                       'sample) or a 2d matrix where rows'\
                                       'correspond to different samples.'
        if query.ndim == 1:
            return tensor_norm(query.dimshuffle('x', 0))
        return tensor_norm(query)

    def _neighbours(self, query):
        """Returns the labels and similarities between keys in memory and query.

        Nearest neighbours are defined as the samples in memory that maximize
        the cosine similarity between itself and a given query sample.
        """

        n_queries = query.shape[0]

        # Reshape the tensors for broadcasting and calculate cosine similarity
        q = query.dimshuffle(0, 'x', 1) # (n_query, 1, key_size)
        M = self.K.dimshuffle('x', 0, 1) # (1, memory_size, key_size)
        similarity = T.sum(q*M, axis=2) # (n_query, memory_size)

        # Find the k-nearest neighbours in terms of cosine similarity.
        k_nbrs = T.argsort(similarity, axis=1)[:, ::-1][:, :self.k_nbrs]

        # Index the labels & similarity matrix to get values for each neighbour.
        # 'repeat' formats a list as: ([0]**n_queries, ... [n-1]*n_queries)
        idx = T.extra_ops.repeat(T.arange(n_queries), self.k_nbrs)
        k_nbrs_y = self.V[k_nbrs.flatten()].reshape(k_nbrs.shape)
        k_nbrs_sim = similarity[idx, k_nbrs.flatten()].reshape(k_nbrs.shape)

        return k_nbrs, k_nbrs_y, k_nbrs_sim
    
    def _loss(self, nbrs, nbrs_y, query, query_y):
        """Builds a theano graph for computing memory triplet loss."""

        def get_idx(q_nbrs, q_mem):
            """Gets the index of sample in memory for computing loss.

            We first look to see if the query label can be found in the 
            retrieved neighbours, and if not, look to memory for a key with
            the same value.

            We keep track of a boolean mask, which indices whether or not we
            were able to find a sample with a label that matches the query.
            """

            # Whether a matching sample can be found in neighbours or memory
            any_match_nbrs = T.any(q_nbrs, axis=1)
            any_match_mem = T.any(q_mem, axis=1)
            any_match = T.or_(any_match_nbrs, any_match_mem)

            # Look in neighbours then memory for corresponding sample.
            # If from neighbours, we need to retrieve the full mem idx.
            rows = T.arange(nbrs.shape[0])
            idx = T.switch(any_match_nbrs, 
                           nbrs[rows, tensor_choose_k(q_nbrs, self.rng, k=1)], 
                           tensor_choose_k(q_mem, self.rng, k=1, random=True))

            return (idx, any_match)

        # Make the labels broadcastable for indexing
        query_y_2d = T.reshape(query_y, (-1, 1))

        query_in_nbrs = T.eq(query_y_2d, nbrs_y) #(n_queries, self.k_nbrs)
        query_in_mem = T.eq(query_y_2d, T.reshape(self.V, (1, -1)))

        positive = get_idx(query_in_nbrs, query_in_mem)
        pos_loss = T.sum(query*self.K[positive[0]], axis=1)*positive[1]

        negative = get_idx(T.invert(query_in_nbrs), T.invert(query_in_mem))
        neg_loss = T.sum(query*self.K[negative[0]], axis=1)*negative[1]

        # Only return the positive components
        return T.maximum(0, neg_loss - pos_loss + self.alpha)


    def _update(self, nbrs, nbrs_y, query, query_y):
        """Builds a graph for performing updates to memory module.

        Two different kinds of updates can be performed, which depend on the
        accuracy of the retrieved neighbours:

        Case 1) If V[n_1] == q, update the key by taking the average of the
                current key and q, and normalizing it.

        Case 2) If V[n_1] != q, find memory items with maximum age, and write
                to one of them (randomly chosen)

        Finally, regardless of the update strategy, increment the age of all
        non-updated indices by 1.
        """

        # Set up the graph for our shared memory variables
        new_K, new_A, new_V = self.K, self.A, self.V

        # Condition (1): First returned neighbour shares the same query label
        correct_query = T.eq(nbrs_y[:, 0], query_y).nonzero()[0]
        correct_mem = nbrs[correct_query, 0] # Idx to memory keys

        normed_keys = tensor_norm(query[correct_query] + new_K[correct_mem])
        new_K = T.set_subtensor(new_K[correct_mem], normed_keys)
        new_A = T.set_subtensor(new_A[correct_mem], 0.)

        # Condition (2): First returned neighbour does not share query label.
        # Add the key and label from query to memory
        incorrect_mask = T.neq(nbrs_y[:, 0], query_y)
        incorrect_query = incorrect_mask.nonzero()[0]

        # We need to find len(incorrect_query) locations in memory to write to.
        # Noise is added to randomize selection.
        age_mask = T.ge(new_A, T.max(new_A) - self.C)
        oldest_idx = tensor_choose_k(age_mask, self.rng, 
                                     k=T.sum(incorrect_mask), 
                                     random=True)

        new_K = T.set_subtensor(new_K[oldest_idx], query[incorrect_query])
        new_V = T.set_subtensor(new_V[oldest_idx], query_y[incorrect_query])
        new_A = T.set_subtensor(new_A[oldest_idx], 0.)

        # Increment the age of all non-updated indices by 1
        update_mask = T.ones_like(new_A, dtype=theano.config.floatX)
        update_mask = T.set_subtensor(update_mask[correct_mem], 0.)
        update_mask = T.set_subtensor(update_mask[oldest_idx], 0.)
        new_A = new_A + update_mask

        return {(self.K, new_K), (self.V, new_V), (self.A, new_A)}

    def build_loss_and_updates(self, query, query_y):
        """Builds theano graphs for loss and updates to memory."""

        normed_query = self._format_query(query)

        # Find the labels and similarity of a query vector to memory
        nbrs, nbrs_y, _ = self._neighbours(normed_query)

        # Build the graph for computing loss and updates to shared memory
        loss = self._loss(nbrs, nbrs_y, normed_query, query_y)
        updates = self._update(nbrs, nbrs_y, normed_query, query_y)

        return loss, updates

    def query(self, query):
        """Queries the memory module for a label to a query."""

        # Normalize the query value.
        normed_query = self._format_query(query)

        # Find the labels and similarity of a query vector to memory
        _, labels, similarity = self._neighbours(normed_query)

        # Return label of the closest match + softmax over retrieved similarities
        return T.cast(labels[:, 0], 'int32'), T.nnet.softmax(self.t*similarity)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    """Iterates through a minibatch of examples (via Lasagne mnist example)."""
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
    """main for testing."""

    # ------------------- Dataset -------------------------------------
    import time
    from sklearn import datasets, preprocessing
    from sklearn.model_selection import train_test_split

    # Dataset
    n_features = 30
    n_samples = 500

    # Learning
    batch_size = 10 # Note, also controls age of memory
    num_epochs = 20

    # Memory
    memory_size = 100
    key_size = n_features
    k_nbrs = 5

    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                        n_informative=5, n_redundant=3,
                                        random_state=42, n_classes=2)

    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)


    # ------------- Memory module --------------------------------------

    # Build the memory module
    query_var = T.matrix('query_var')
    query_target = T.ivector('query_target')

    memory = MemoryModule(memory_size, key_size, k_nbrs)
    loss, updates = memory.build_loss_and_updates(query_var, query_target)
    query_lbls, query_sim = memory.query(query_var)


    # Compile theano functions to test graph
    train_fn = theano.function([query_var, query_target], loss, updates=updates,
                               allow_input_downcast=True)

    query_fn = theano.function([query_var], [query_lbls, query_sim], allow_input_downcast=True)

    #sys.exit(1)

    import time
    print "Starting training..."
    for epoch in range(num_epochs):

        start = time.time()
        train_acc = 0
        train_batches = 0
        # In each epoch, we do a full pass over the training data:
        for batch in iterate_minibatches(X_train[:10], y_train[:10],
                                         batchsize=10, shuffle=True):
            inputs, targets = batch

            pred_loss = train_fn(inputs, targets)

            labels, sim = query_fn(inputs)
            train_acc += np.mean(labels == targets)
            train_batches += 1 

            print 'pred_loss: \n', pred_loss
            print 'Targets:   \n', targets


        valid_batches = 0
        valid_acc = 0.
        for batch in iterate_minibatches(X_val, y_val, batch_size, shuffle=False):
            inputs, targets = batch

            labels, sim = query_fn(inputs)

            #print 'Valid labels: \n', targets
            #print 'Predc labels: \n', labels

            valid_acc += np.mean(labels == targets)
            valid_batches += 1

        print 'Epoch %d took: %2.4fs'%(epoch, time.time() - start)
        print '  Train Accuracy: %2.4f'%(train_acc / train_batches)
        print '  Valid Accuracy: %2.4f'%(valid_acc / valid_batches)

    used_memory_mask = memory.V.get_value() != -1
    unused_memory_mask = memory.V.get_value() == -1
    print 'Percentage of used memory:    %2.6f'%(float(np.sum(used_memory_mask)) / float(memory_size))
    print 'Percentage of un-used memory: %2.6f'%(float(np.sum(unused_memory_mask)) / float(memory_size))

    '''
    import matplotlib.pyplot as plt
    used_mem_age = memory.A.get_value()[used_memory_mask]
    used_age_hist = [np.sum(used_mem_age == u) for u in np.arange(50)]
    used_age_hist = np.vstack(used_age_hist)

    plt.hist(memory.A.get_value())
    plt.show()
    '''

if __name__ == '__main__':
    main()
