import os, sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as nn

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

def tensor_norm(X, keepdims=True):
    return X / T.sqrt(T.sum(X**2, axis=1, keepdims=keepdims))

def norm(X, keepdims=True):
    return X / np.sqrt(np.sum(X**2, axis=1, keepdims=keepdims))

def int32x(x):
    return T.cast(x, 'int32')

class MemoryModule(object):
    """Implements a memory module using theano.

    Note: It is important that every key is initialized differently.
          If all keys have the same unit norm, the module will fail to update
          properly as each query will produce the same nearest neighbours.
    """

    def __init__(self, mem_size, key_size, k_nbrs, alpha=0.1, t=40, seed=1234,
                 eps=1e-8, K=None, V=None, A=None):

        self.mem_size = mem_size
        self.key_size = key_size
        self.k_nbrs = k_nbrs
        self.alpha = alpha
        self.t = t # Softmax temperature
        self.rng = RandomStreams(seed=seed)
        self.eps = eps

        if K is None:
            K = norm(np.random.randn(mem_size, key_size))
            K = K.astype(theano.config.floatX)
        else:
            assert np.allclose(np.sum(K**2, axis=1), np.ones(K.shape[0])), \
                   'The rows of K must be unit norm'

        if V is None:
            V = np.ones(mem_size, dtype=theano.config.floatX)*-1
        if A is None:
            A = np.zeros(mem_size, dtype=theano.config.floatX)

        # Memory keys (K), Values (V), and Ages (A)
        self.K = theano.shared(K, name='keys')
        self.V = theano.shared(V, name='values')
        self.A = theano.shared(A, name='ages')

    def _neighbours(self, query):
        """Returns the labels and similarities between keys in memory and query.

        Nearest neighbours are defined as the samples in memory that maximize
        the cosine similarity between itself and a given query sample.
        """

        n_queries = query.shape[0]

        # Reshape the tensors for broadcasting and calculate cosine similarity
        q = query.dimshuffle(0, 'x', 1) # (n_query, 1, key_size)
        M = self.K.dimshuffle('x', 0, 1) # (1, memory_size, key_size)
        cosine_sim = T.sum(q*M, axis=2) # (n_query, memory_size)

        # Find the k-nearest neighbours in terms of cosine similarity.
        k_nearest = T.argsort(cosine_sim, axis=1) # Ascending order
        k_nearest = k_nearest[:, ::-1][:, :self.k_nbrs] # Descending order

        # Find the calculated cosine similarity for the nearest neighbours
        # Note that repeats formats as: ([0]**n_queries, ... [n-1]*n_queries)
        idx = T.extra_ops.repeat(T.arange(n_queries), self.k_nbrs)
        k_similar = cosine_sim[idx, k_nearest.flatten()]
        k_similar = k_similar.reshape(k_nearest.shape)

        # Find the label of the closest neighbour for each query
        k_labels = self.V[k_nearest.flatten()].reshape(k_nearest.shape)

        # Return the indices, labels at the indices, cosine similarity at indices
        return k_nearest, k_labels, k_similar

    def _loss(self, nbrs, nbrs_labels, query, query_labels):
        """Builds a theano graph for computing memory module loss.

        TODO: In scan_fn, when accessing memory for a sample with the same label
              as query, make sure the returned indices are randomized.
              Currently this is not implemented as there are issues when only a
              single element of a class exists in memory (incompatable theano types)
        """

        # Dummy variables for selecting postive / negative neighbour
        CONST_TRUE = T.constant(1.0, dtype='int32')
        CONST_FALSE = T.constant(0.0, dtype='int32')

        def scan_fn(nbrs_idx, nbrs_y, query_y):
            """Returns the positive and negative sample for a query."""

            def get_sample_idx(compare_op):

                assert compare_op in [T.eq, T.neq]

                # See if the neighbours contain a label, if not check memory
                nbr_match = compare_op(nbrs_y, query_y)
                mem_match = compare_op(self.V, query_y)

                # Whether a matching sample can be found in neighbours or memory
                mask = ifelse(T.any(nbr_match), CONST_TRUE,
                        ifelse(T.any(mem_match), CONST_TRUE, CONST_FALSE))

                # An index to the full memory (using either a retrieved
                # neighbour, or random sample in memory) of a matching sample
                inbr = int32x(nbr_match.nonzero()[0][0])
                imem = int32x(mem_match.nonzero()[0][0])

                # Whether the match is found in the neighbours or memory
                match = ifelse(T.any(nbr_match), int32x(nbrs_idx[inbr]),
                         ifelse(T.any(mem_match), imem, CONST_FALSE))

                return match, mask

            pos_idx, pos_mask = get_sample_idx(T.eq)
            neg_idx, neg_mask = get_sample_idx(T.neq)

            return (pos_idx, neg_idx, pos_mask, neg_mask)


        results, updates = theano.scan(scan_fn,
                    sequences=[nbrs, nbrs_labels, query_labels])

        pos_idx, neg_idx, pos_mask, neg_mask = results

        pos_mask = pos_mask.dimshuffle(0, 'x')
        neg_mask = neg_mask.dimshuffle(0, 'x')
        negative_loss = T.sum(query*self.K[neg_idx]*neg_mask, axis=1)
        positive_loss = T.sum(query*self.K[pos_idx]*pos_mask, axis=1)

        # Relu trick taken from TF implementation
        loss = T.nnet.relu(negative_loss - positive_loss + self.alpha) - self.alpha

        return loss

    def _update(self, nbrs, nbrs_labels, query, query_label):
        """Builds a graph for performing updates to memory module.

        Two different kinds of updates can be performed, which depend on the
        accuracy of the retrieved neighbours:

        Case 1) If V[n_1] == q, update the key by taking the average of the
                current key and q, and normalizing it.
        Case 2) If V[n_1] != q, find memory items with maximum age, and write
                to one of them (randomly chosen)

        Finally, (regardless of the update strategy) increment the age of all
        non-updated indices by 1.
        """

        # Set up the graph for our shared memory variables
        new_K, new_A, new_V = self.K, self.A, self.V

        # Condition (1)
        # Note that we need the slice to force an (n x 1) matrix for comparison
        correct_query = T.eq(nbrs_labels[:, 0:1], query_label).nonzero()[0]
        correct_memory = nbrs[correct_query, 0] # Idx to memory keys

        normed_keys = tensor_norm(query[correct_query] + new_K[correct_memory])
        new_K = T.set_subtensor(new_K[correct_memory], normed_keys)
        new_A = T.set_subtensor(new_A[correct_memory], 0)

        # Condition (2)
        # Find where the first returned neighbour does not share query label
        incorrect_mask = T.neq(nbrs_labels[:, 0:1], query_label)
        incorrect_query = incorrect_mask.nonzero()[0]
        incorrect_memory = nbrs[incorrect_query, 0]

        # Depending on how many times the first retrieved neighbour did not match
        # the query, we need to find that many spots in memory (with maximum age)
        # to write the sample to. Note that random noise is added to selection
        age_mask = T.eq(new_A, T.max(new_A))
        noise = self.rng.uniform((self.mem_size,), 0, self.mem_size)
        oldest_idx = T.argsort(age_mask*noise)[::-1] # sorted decreasing
        oldest_idx = oldest_idx[:T.sum(incorrect_mask)]

        # Note that query_label is reshaped to an (n x 1) matrix
        new_K = T.set_subtensor(new_K[incorrect_memory], query[incorrect_query])
        new_V = T.set_subtensor(new_V[incorrect_memory], query_label[incorrect_query, 0])
        new_A = T.set_subtensor(new_A[incorrect_memory], 0.)

        # Increment the age of all non-updated indices by 1
        all_update_mask = T.ones_like(new_A)
        all_update_mask = T.set_subtensor(all_update_mask[correct_memory], 0)
        all_update_mask = T.set_subtensor(all_update_mask[incorrect_memory], 0)
        new_A = new_A + all_update_mask

        return [(self.K, new_K), (self.V, new_V), (self.A, new_A)]

    def build_loss_and_updates(self, query, query_labels):

        # Find the labels and similarity of a query vector to memory
        normed_query = tensor_norm(query)

        if query_labels.ndim == 1:
            query_labels = T.reshape(query_labels, (-1, 1))

        # Find the labels and similarity of a query vector to memory
        nbrs, nbrs_labels, _ = self._neighbours(normed_query)

        # Build the graph for computing loss updates to shared memory
        loss = self._loss(nbrs, nbrs_labels, normed_query, query_labels)
        updates = self._update(nbrs, nbrs_labels, normed_query, query_labels)

        return loss, updates

    def query(self, query):
        """Queries the memory module for a label to a query."""

        # Normalize the query value.
        normed_query = tensor_norm(query)

        # Find the labels and similarity of a query vector to memory
        _, labels, similarity = self._neighbours(normed_query)

        # Return label of the closest match + softmax over retrieved similarities
        return labels[:, 0], T.nnet.softmax(self.t*similarity)


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

    # ------------------- Dataset -------------------------------------
    import time
    from sklearn import datasets, preprocessing
    from sklearn.model_selection import train_test_split

    # Dataset
    n_features = 30
    n_classes = 2

    # Learning
    batch_size = 50 # Note, also controls age of memory
    num_epochs = 50
    n_samples = 1000

    # Memory
    memory_size = 1000
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

    # Compile theano functions to test graph
    train_fn = theano.function([query_var, query_target], loss, updates=updates,
                        allow_input_downcast=True, on_unused_input='ignore')


    num_epochs = 1
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 100, shuffle=True):
            inputs, targets = batch

            pred_loss = train_fn(inputs, targets)


        print 'Memory in use: ', np.sum(memory.A.get_value() < np.max(memory.A.get_value()))
        unique = list(set(memory.A.get_value()))

        for u in unique:
            print '%s: %d'%(u, np.sum(memory.A.get_value() == u))


        import matplotlib.pyplot as plt

        used_memory_mask = memory.V.get_value() != -1
        unused_memory_mask = memory.V.get_value() == -1

        used_mem_age = memory.A.get_value()[used_memory_mask]
        used_age_hist = [np.sum(used_mem_age == u) for u in np.arange(50)]
        used_age_hist = np.vstack(used_age_hist)

        plt.hist(used_age_hist, np.arange(50))

        print 'Percentage of used memory:    ',float(np.sum(used_memory_mask)) / float(memory_size)
        print 'Percentage of un-used memory: ',float(np.sum(unused_memory_mask)) / float(memory_size)
        plt.show()

if __name__ == '__main__':
    main()
