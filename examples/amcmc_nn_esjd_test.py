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
# (X_t) => (Pr[X_{t+1}[0]], S[1:n-1])
# (X_t, X_{t+1}[0], S[1:n-1]) => (Pr[X_{t+1}[1]], S[2:n-1])
#  ...
# (X_t, X_{t+1}[0:i-1], S[i:n-1]) => (Pr[X_{t+1}[i]], S[i+1:n-1])
#  ...
# (X_t, X_{t+1}[0:n-2], S[n-1]) => (Pr[X_{t+1}[n-1]])
#
# Each `small network' is input->[tanh]->[softmax for Pr, sigmoid for S]->output
# and n_hidden = [2/3 * (n_in + n_out)] # or max(2/3 * n_in, n_out)?

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

# H represents hidden layer

rng = np.random.RandomState(1234)
srng = RandomStreams(1234)

x = T.ivector('x')
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
h_0 = T.tanh(T.dot(x, w_00) + b_00)
y_0 = T.dot(h_0, w_01) + b_01
distn_0 = T.nnet.softmax(y_0[:-1])
s_1 = T.nnet.sigmoid(y_0[-1:])
cate_chosen_0 = T.flatten(srng.multinomial(n = 1, pvals = distn_0))
ind_chosen_0 = T.argmax(cate_chosen_0)

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
h_1 = T.tanh(T.dot(T.concatenate([cate_chosen_0, s_1]), w_10) + b_10)
distn_1 = T.nnet.softmax((T.dot(h_1, w_11) + b_11))
cate_chosen_1 = T.flatten(srng.multinomial(n = 1, pvals = distn_1))
ind_chosen_1 = T.argmax(cate_chosen_1)

prob = distn_0[0, ind_chosen_0] * distn_1[0, ind_chosen_1]
activate = theano.function(
    [x], [ind_chosen_0, ind_chosen_1, prob],
)

# Training
mx = T.imatrix('mx')
proposals = T.imatrix('proposals')
esjqs = T.ivector('esjqs')
target = esjqs
arange = T.arange(target.shape[0])

mh_0 = T.tanh(T.dot(mx, w_00) + b_00)
my_0 = T.dot(mh_0, w_01) + b_01
mdistn_0 = T.nnet.softmax(my_0[arange, :-1])
ms_1 = T.nnet.sigmoid(my_0[arange, -1:])
target = target * mdistn_0[arange, proposals[arange, 0]]
cates_0 = T.extra_ops.to_one_hot(proposals[arange, 0], 3)
inp_1 = T.concatenate([cates_0, ms_1], axis = 1)
mh_1 = T.tanh(T.dot(inp_1, w_10) + b_10)
my_1 = T.dot(mh_1, w_11) + b_11
mdistn_1 = T.nnet.softmax(my_1)
target = target * mdistn_1[arange, proposals[arange, 1]]
avg_target = T.mean(target)
log_avg_target = T.log(avg_target)

weights = [w_00, w_01, w_10, w_11]
g_weights = [T.grad(log_avg_target, w) for w in weights]
update = theano.function(
    [mx, proposals, esjqs],
    [log_avg_target],
    updates = [
        (w, w + 0.1 * gw)
        for w, gw in zip(weights, g_weights)
    ],
)

# Probing
prop_prob = distn_0[0, ind_chosen_0] * distn_1[0, ind_chosen_1]
proposal_prob = theano.function(
    [mx, proposals],
    [target],
    givens = {esjqs: T.ones((mx.shape[0],), dtype = 'int32')}
)

# Sample run
# >>> activate([0,0,1,1,0,0])
# [array(1), array(1), array(0.4253576812753404)]
# >>> proposal_prob([[0,0,1,1,0,0]], [[2,1]])
# [array([ 0.00322697])]
# >>> update([[0,0,1,1,0,0],[0,0,1,1,0,0]],[[1,1],[2,0]],[0,3])
# [array(-5.365478170295351)]
# >>> proposal_prob([[0,0,1,1,0,0],[0,0,1,1,0,0]], [[1,1],[2,1]])
# [array([ 0.07009229,  0.03638326])]

