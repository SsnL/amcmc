import numpy as np
import theano
import theano.tensor as T

# A small test of achieveability for AMCMC_NN_ESJD
# Model copied from amcmc_nn_esjd.py as below:

# An M-H method

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