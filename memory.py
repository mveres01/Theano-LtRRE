"""Main script that defines the memory module."""

from collections import  OrderedDict
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def norm(X, eps=1e-8, keepdims=True):
    """Returns an l2-normalized version of an input along each row."""
    return X / (np.sqrt(np.sum(X**2, axis=1, keepdims=keepdims)) + eps)


def tensor_norm(X, eps=1e-8, keepdims=True):
    """Returns an l2-normalized version of a theano tensor along each row."""
    return X / (T.sqrt(T.sum(X**2, axis=1, keepdims=keepdims)) + eps)


def tensor_choose_k(boolean_mask, rng, k=1, random=False):
    """Selects k elements from each row of boolean_mask.

    Parameters
    ----------
    boolean_mask: A 2d Boolean matrix
      Each row is a sample, and each column with True is a possible choice
    rng: An instance of theano.sandbox.rng_mrg.MRG_RandomStreams
    k: An integer denoting the number of samples to choose
    random: Boolean flag indicating whether to select 'k' indices in order
        (random=False), or whether to randomly choose 'k' indices
        (randome=True).
    """

    mask = boolean_mask
    if mask.ndim > 2:
        raise Exception('Input tensor must be either 1d or 2d.')
    elif mask.ndim == 1:
        mask = mask.dimshuffle('x', 0)

    assert T.lt(k, mask.shape[1]), 'k must be < then # of possible choices'

    if random is True:
        noise = rng.uniform(mask.shape, low=0, high=mask.shape[1])
    else:
        noise = T.arange(mask.shape[1])[::-1] + 1 # Descending order
        noise = noise.dimshuffle('x', 0)

    if k == 1:
        return T.argmax(mask*noise, axis=1)
    return T.argsort(mask*noise, axis=1)[:, ::-1][:, :k]

def tensor_format_query(query):
    """Convenience function for formatting query vector / matrix."""

    if not T.le(query.ndim, 2):
        raise ValueError('Query must either be 1d (single sample) or a '\
                         '2d matrix where rows are different samples.')

    if query.ndim == 1:
        return tensor_norm(query.dimshuffle('x', 0))
    return tensor_norm(query)


class MemoryModule(object):
    """Implements a memory module using theano.

    Note: Assumes initial class labels of -1
    Note: When training with neural networks, make sure that the size of memory
          is larger then the batch size, or this module will throw an error when
          trying find too many indexs' to memory.
    """

    def __init__(self, mem_size, key_size, k_nbrs, alpha=0.1, t=40, C=3,
                 seed=1234, K=None, V=None, A=None):

        assert k_nbrs <= mem_size, 'k_nbrs must be less then (or equal) mem size'

        self.mem_size = mem_size
        self.key_size = key_size
        self.k_nbrs = k_nbrs
        self.alpha = alpha
        self.t = t # Softmax temperature
        self.C = C # Threshold for sampling mamx ages during update
        self.rng = RandomStreams(seed=seed)

        if K is None:
            K = np.random.randn(mem_size, key_size)
            K = norm(K).astype(theano.config.floatX)
        assert np.allclose(np.sum(K**2, axis=1), 1.), \
            'Supplied K must be unit norm.'
        self.K = theano.shared(K, name='keys')

        if V is None:
            V = np.full(mem_size, -1, dtype='int32')
        self.V = theano.shared(V, name='values')

        if A is None:
            A = np.full(mem_size, C+1, dtype=theano.config.floatX)
        self.A = theano.shared(A, name='ages')

    def _neighbours(self, query):
        """Returns the labels and similarities between keys in memory and query.

        Nearest neighbours are defined as the samples in memory that maximize
        the cosine similarity between itself and a given query sample.
        """

        # Because the query and memory keys are aready normalized, cosine
        # similarity can be calculated through a single matrix multiplication.
        similarity = T.dot(query, self.K.T)

        # Find the k-nearest neighbours
        k_nbrs = T.argsort(similarity, axis=1)[:, ::-1][:, :self.k_nbrs]
        k_nbrs_y = self.V[k_nbrs.flatten()].reshape(k_nbrs.shape)

        # Make a pseude row index via repeat
        idx = T.extra_ops.repeat(T.arange(query.shape[0]), self.k_nbrs)
        k_nbrs_sim = similarity[idx, k_nbrs.flatten()].reshape(k_nbrs.shape)

        return k_nbrs, k_nbrs_y, k_nbrs_sim

    def _loss(self, nbrs, nbrs_y, query, query_y):
        """Builds a theano graph for computing memory triplet loss."""

        def get_idx(q_nbrs, q_mem):
            """Gets the index of sample in memory for computing loss.

            We first look to see if the query label can be found in the
            retrieved neighbours, and if not, look to memory for a key with
            the same value.

            We keep track of a boolean mask, which indicates whether or not we
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
        age_mask = T.ge(new_A, T.max(new_A) - self.C) #1d
        oldest_idx = tensor_choose_k(age_mask, self.rng,
                                     k=T.sum(incorrect_mask),
                                     random=True).flatten()

        new_K = T.set_subtensor(new_K[oldest_idx], query[incorrect_query])
        new_V = T.set_subtensor(new_V[oldest_idx], query_y[incorrect_query])
        new_A = T.set_subtensor(new_A[oldest_idx], 0.)

        # Increment the age of all non-updated indices by 1
        new_A = new_A + 1.
        new_A = T.inc_subtensor(new_A[correct_mem], -1.)
        new_A = T.inc_subtensor(new_A[oldest_idx], -1.)

        return OrderedDict({(self.K, new_K), (self.V, new_V), (self.A, new_A)})

    def build_loss_and_updates(self, query, query_y):
        """Builds theano graphs for loss and updates to memory."""

        normed_query = tensor_format_query(query)

        # Find the indices and labels of memory to a query
        nbrs, nbrs_y, _ = self._neighbours(normed_query)

        # Build the graph for computing loss and updates to shared memory
        loss = self._loss(nbrs, nbrs_y, normed_query, query_y)
        updates = self._update(nbrs, nbrs_y, normed_query, query_y)

        return loss, updates

    def get_memory(self):
        """Returns the current stored memory."""
        return (self.K.get_value(), self.V.get_value(), self.A.get_value())

    def query(self, query_in):
        """Queries the memory module for a label to a sample."""

        # Make sure all inputs are normalized, so future math holds
        normed_query = tensor_format_query(query_in)

        _, labels, similarity = self._neighbours(normed_query)

        # Return label of the closest match + softmax over retrieved similarities
        return T.cast(labels[:, 0], 'int32'), T.nnet.softmax(self.t*similarity)

    def set_memory(self, K, V, A):
        """Sets the current memory."""

        if not  all(s.shape[0] == self.mem_size for s in [K, V, A]):
            raise ValueError('Memory items must have %d samples.'%self.mem_size)
        if not np.allclose(np.sum(K**2, axis=1), 1.):
            raise ValueError('The rows of K must be unit norm')

        self.K.set_value(K)
        self.V.set_value(V)
        self.A.set_value(A)
