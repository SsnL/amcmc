import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# A small test of achieveability for AMCMC_NN_ESJD
# Model copied from ../inference/amcmc_nn_esjd.py as below:

# Proposal model for n RVs [A network of n `small network's]:
#
# [Each of the following row is a `small network']
# [inclusive indices]
#
# (X_t) => (Pr[X_{t+1}[0]], H[1:n-1])
# (X_t, X_{t+1}[0], H[1:n-1]) => (Pr[X_{t+1}[1]], H[2:n-1])
#  ...
# (X_t, X_{t+1}[0:i-1], H[i:n-1]) => (Pr[X_{t+1}[i]], H[i+1:n-1])
#  ...
# (X_t, X_{t+1}[0:n-2], H[n-1]) => (Pr[X_{t+1}[n-1]])
#
# Each `small network' is input->[tanh]->[softmax for Pr, sigmoid for H]->output
# and n_hidden = [2/3 * (n_in + n_out)]

# Training
# 1. Target: maximizing ESJD
# 2. Quantify target:
#      each t, sample {m} proposals with input X_t for ESJD estimation in training
# 3. Training at every {train_int} MCMC iterations in batches
# 4. Normalized with {l1_reg} * L1 and {l2_leg} * L2 norms

# Implemented with Theano

# ====

# Toy
# 2 RVs, each with 3 values
# 6 => 6 => 3+1
# 3+1 => 4 => 3

# S is H above, H is hidden layer instead

rng = np.random.RandomState(1234)
srng = RandomStreams(1234)

x_t = T.ivector('x_t')
w_00 = theano.shared(
    value = np.asarray(
        rng.uniform(
            low = -np.sqrt(6. / 12),
            high = np.sqrt(6. / 12),
            size = (6, 6)
        ),
        dtype = theano.config.floatX
    ),
    name = 'w_00',
    borrow = True,
)
b_00 = theano.shared(
    value = np.zeros((6,), dtype=theano.config.floatX),
    name = 'b_00',
    borrow = True,
)
w_01 = theano.shared(
    value = 4 * np.asarray(
        rng.uniform(
            low = -np.sqrt(6. / 10),
            high = np.sqrt(6. / 10),
            size = (6, 4)
        ),
        dtype = theano.config.floatX
    ),
    name = 'w_01',
    borrow = True,
)
b_01 = theano.shared(
    value = np.zeros((4,), dtype=theano.config.floatX),
    name = 'b_01',
    borrow = True,
)
h_0 = T.tanh(T.dot(x_t, w_00) + b_00)
y_0 = T.dot(h_0, w_01) + b_01
pp_0 = T.nnet.softmax(y_0[:-1])
s_1 = y_0[-1]
cp_0 = T.flatten(srng.multinomial(n = 1, pvals = pp_0))
p_0 = T.argmax(cp_0)

w_10 = theano.shared(
    value = np.asarray(
        rng.uniform(
            low = -np.sqrt(6. / 8),
            high = np.sqrt(6. / 8),
            size = (4, 4)
        ),
        dtype = theano.config.floatX
    ),
    name = 'w_10',
    borrow = True,
)
b_10 = theano.shared(
    value = np.zeros((4,), dtype=theano.config.floatX),
    name = 'b_10',
    borrow = True,
)
w_11 = theano.shared(
    value = 4 * np.asarray(
        rng.uniform(
            low = -np.sqrt(6. / 7),
            high = np.sqrt(6. / 7),
            size = (4, 3)
        ),
        dtype = theano.config.floatX
    ),
    name = 'w_11',
    borrow = True,
)
b_11 = theano.shared(
    value = np.zeros((3,), dtype=theano.config.floatX),
    name = 'b_11',
    borrow = True,
)
h_1 = T.tanh(T.dot(T.concatenate([cp_0, [s_1]]), w_10) + b_10)
pp_1 = T.nnet.softmax((T.dot(h_1, w_11) + b_11))
cp_1 = srng.multinomial(n = 1, pvals = pp_1)
p_1 = T.argmax(cp_1)

prob = pp_0[0, p_0] * pp_1[0, p_1]
activate = theano.function(
    [x_t],
    [p_0, p_1, prob],
)

# Training for one datum
prop = T.ivector('p')
esjq = T.iscalar('esjq')
p_prob = pp_0[0, T.argmax(prop[:3])] * pp_1[0, T.argmax(prop[3:])]
weights = [w_00, w_01, w_10, w_11]
g_weights = [T.grad(p_prob, w) for w in weights]

update = theano.function(
    [x_t, prop, esjq],
    [],
    updates = [
        (w, w + 0.1 * esjq * gw)
        for w, gw in zip(weights, g_weights)
    ],
)

# Sample run
# >>> activate([0,0,1,1,0,0])
# [array(1), array(1), array(0.3294365079931716)]
# >>> activate([0,0,1,1,0,0])
# [array(1), array(2), array(0.5459377730141635)]
# >>> update([0,0,1,1,0,0], [0,1,0,0,1,0], 1)
# []
# >>> activate([0,0,1,1,0,0])
# [array(1), array(1), array(0.45126529550025507)]
# >>> update([0,0,1,1,0,0], [0,1,0,0,1,0], 1)
# []
# >>> update([0,0,1,1,0,0], [0,1,0,0,1,0], 1)
# []
# >>> update([0,0,1,1,0,0], [0,1,0,0,1,0], 1)
# []
# >>> activate([0,0,1,1,0,0])
# [array(1), array(1), array(0.7385558686425071)]
# >>> activate([0,0,1,1,0,0])
# [array(1), array(2), array(0.21173994572840332)]
# >>> update([0,0,1,1,0,0], [0,1,0,0,0,1], 1)
# []
# >>> update([0,0,1,1,0,0], [0,1,0,0,0,1], 10)
# []
# >>> activate([0,0,1,1,0,0])
# [array(1), array(2), array(0.878903229111633)]
# >>> update([0,0,1,1,0,0], [0,1,0,0,0,1], 10)
# []
# >>> activate([0,0,1,1,0,0])
# [array(1), array(2), array(0.9484709462065333)]

