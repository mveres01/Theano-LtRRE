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
    """Implements a memory module using theano."""

    def __init__(self, mem_size, key_size, k_nbrs, alpha=0.1, t=40, seed=1234,
                 eps=1e-8):

        # Memory keys (K), Values (V), and Ages (A)
        self.K = theano.shared(norm(np.random.randn(mem_size, key_size)))
        self.V = theano.shared(np.ones(mem_size)*-1)
        self.A = theano.shared(np.zeros(mem_size))
        self.mem_size = mem_size
        self.key_size = key_size
        self.k_nbrs = k_nbrs
        self.alpha = alpha
        self.t = t # Softmax temperature
        self.rng = RandomStreams(seed=seed)
        self.eps = eps

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

    def _loss(self, query, query_labels):
        """Builds a theano graph for computing memory module loss.

        NOTE: This function is called indirectly, and assumes that the input
        query is already normalized
        """

        # Dummy variables for selecting postive / negative neighbour
        CONST_TRUE = T.constant(1.0, dtype='int32')
        CONST_FALSE = T.constant(0.0, dtype='int32')

        def scan_fn(k_nbrs, k_nbrs_y, query_y):
            """Returns the positive and negative sample for a single query.

            TODO: When accessing memory for a sample with the same label, we
                  should make sure the returned indices are randomized. Currently
                  this is not implemented as there are issues when only a single
                  element of a class exists in memory (incompatable theano types)
            """

            def get_sample_idx(compare_op):

                assert compare_op in [T.eq, T.neq]

                # See if the neighbours contain a label, if not check memory
                nbr_match = compare_op(k_nbrs_y, query_y)
                mem_match = compare_op(self.V, query_y)

                # Whether a matching sample can be found in neighbours or memory
                valid_idx = ifelse(T.any(nbr_match), CONST_TRUE,
                            ifelse(T.any(mem_match), CONST_TRUE, CONST_FALSE))

                # An index to the full memory (either a retrieved neighbour, or
                # random sample in memory) of a matching sample
                nbr_idx = int32x(nbr_match.nonzero()[0][0])
                mem_idx = int32x(mem_match.nonzero()[0][0])

                full_idx = ifelse(T.any(nbr_match), int32x(k_nbrs[nbr_idx]),
                           ifelse(T.any(mem_match), mem_idx, CONST_FALSE))

                return full_idx, valid_idx

            pos_idx, valid_idx = get_sample_idx(T.eq)
            neg_idx, _ = get_sample_idx(T.neq)

            return (valid_idx, pos_idx, neg_idx)

        # Find the labels and similarity of a query vector to memory
        nbrs, labels, similarity = self._neighbours(query)

        (valid_idx, pos_idx, neg_idx), upd = theano.scan(scan_fn,
                                    sequences=[nbrs, labels, query_labels])

        valid_idx = valid_idx.dimshuffle(0, 'x')
        negative_loss = T.sum(query*self.K[neg_idx], axis=1)
        positive_loss = T.sum(query*self.K[pos_idx]*valid_idx, axis=1)

        # Relu trick taken from TF implementation
        loss = T.nnet.relu(negative_loss - positive_loss + self.alpha) - self.alpha

        return negative_loss, pos_idx, neg_idx

    def build_loss(self, query, query_labels):
        # Find the labels and similarity of a query vector to memory
        normed_query = tensor_norm(query)

        if query_labels.ndim == 1:
            query_labels = T.reshape(query_labels, (-1, 1))

        return self._loss(normed_query, query_labels)

    # TODO: Clean up this section
    # ******************************************
    def update(self, query, query_label):
        """Performs updates on the memory module.

        Two different kinds of updates are performed, which depend on the
        accuracy of the retrieved neighbours:

        Case 1) If V[n_1] == q, update the key by taking the average of the
                current key and q, and normalize it.
        Case 2) If V[n_1] != q, find memory items with maximum age, and write
                to one of them (randomly chosen)

        Finally, regardless of the update strategy, increment the age of all
        non-updated indices by 1.
        """

        # Find the labels and similarity of a query vector to memory
        nbrs, labels, similarity = self._neighbours(query)

        # Set up the graph for our shared memory variables
        new_K, new_A, new_V = self.K, self.A, self.V

        # Condition (1)
        correct_query = T.eq(labels[:, 0], query_label).nonzero()[0]
        correct_memory = nbrs[correct_query, 0] # Idx to memory keys

        normed_keys = tensor_norm(query[correct_query] + new_K[correct_memory])
        new_K = T.set_subtensor(new_K[correct_memory], normed_keys)
        new_A = T.set_subtensor(new_A[correct_memory], 0)

        # Condition (2)
        # Find where the first returned neighbour does not share query label
        incorrect_mask = T.neq(labels[:, 0], query_label)
        incorrect_query = incorrect_mask.nonzero()[0]
        incorrect_memory = nbrs[incorrect_query, 0]

        # Find the memory locations with maxmimum age.
        maximum_age_mask = T.eq(new_A, T.max(new_A))

        # Add noise to the returned locations (following method in the paper)
        random_idx = self.rng.uniform((self.mem_size,), 0, self.mem_size)

        # Sample the indices corresponding to the number of mistakes we made
        oldest_idx = T.argsort(maximum_age_mask*random_idx) # sorted increasing
        oldest_idx = oldest_idx[::-1][:T.sum(incorrect_mask)] # sorted decreasing

        new_K = T.set_subtensor(new_K[incorrect_memory], query[incorrect_query])
        new_V = T.set_subtensor(new_V[incorrect_memory], query_label[incorrect_query])
        new_A = T.set_subtensor(new_A[incorrect_memory], 0.)

        # Increment the age of all non-updated indices by 1
        all_update_mask = T.ones_like(new_A)
        all_update_mask = T.set_subtensor(all_update_mask[correct_memory], 0)
        all_update_mask = T.set_subtensor(all_update_mask[incorrect_memory], 0)
        new_A = new_A + all_update_mask

        return new_K, new_V, new_A

    def query(self, query):
        """Queries the memory module for nearest neighbours to a sample."""

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

    # Create dataset of classification task with many redundant and few
    # informative features
    n_features = 30
    n_classes = 2
    num_epochs = 50
    n_samples = 10000

    X, y = datasets.make_classification(n_samples=n_samples, n_features=n_features,
                                        n_informative=5, n_redundant=3,
                                        random_state=42, n_classes=2)

    X_train, X_val, y_train, y_val = \
        train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)


    # ------------- Memory module --------------------------------------

    memory_size = 100
    key_size = n_features
    k_nbrs = 5

    # Build the memory module
    query_var = T.matrix('query_var')
    query_target = T.ivector('query_target')

    memory = MemoryModule(memory_size, key_size, k_nbrs)
    #memory_loss = memory.build_loss(query_var, query_target)
    keys, values, ages = memory.update(query_var, query_target)
    # Compile theano functions to test graph
    update_memory = theano.function(
                        [query_var, query_target],
                        [keys, values, ages],
                        updates=[(memory.K, keys), (memory.V, values), (memory.A, ages)],
                        allow_input_downcast=True, on_unused_input='ignore')

    '''
    labels, pos_idx, neg_idx = memory.build_loss(query_var, query_target)
    memory_loss = theano.function([query_var, query_target],
                            [labels, pos_idx, neg_idx],
                            allow_input_downcast=True, on_unused_input='ignore')
    '''

    num_epochs = 1
    print("Starting training...")
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 10, shuffle=True):
            inputs, targets = batch
            #memory.update(inputs, targets)

            '''
            lb, pi, ni = memory_loss(inputs, targets)
            train_batches += 1

            print 'Labels: \n', lb
            print 'Positive index: \n', pi
            print 'Negative index: \n', ni
            sys.exit(1)
            '''

            lb, pi, ni = update_memory(inputs, targets)
            print 'Keys:  \n', lb
            #print 'Values: \n', pi
            print 'Ages: \n', ni


if __name__ == '__main__':
    main()
