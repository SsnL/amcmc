from ..structure import *
from ..utils import *
from operator import add
from functools import reduce
from itertools import chain
from math import ceil

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# TODO

# An M-H method

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
# A toy example of how the model works is in ../examples/amcmc_nn_esjd_test.py

class AMCMC_NN_ESJD(MCMC):
    def __init__(self, problem, \
            verbose_int = 100, N = 1000, T = 10000, record_start = 3000, \
            train_int = 100, train_steps = 1000, train_batch_size = 5, \
            train_alpha_fn = lambda t: 600.0 / (600 + t)):
        MCMC.__init__(self, \
            problem, "AMCMC_NN_ESJD", verbose_int, N, T, record_start)
        self.train_steps = train_steps
        self.train_int = train_int
        self.train_batch_size = train_batch_size
        self.train_alpha_fn = train_alpha_fn

    def additional_args_str(self):
        return '''training interval: every {interval} iteration(s)
training steps: {steps} step(s)
training batch size: {batch_size}'''.format( \
            interval = self.train_int, steps = self.train_steps, \
            batch_size = self.train_batch_size)

    def get_rv_index(self, rv, val):
        if not hasattr(self, 'rv_index_mem'):
            self.rv_index_mem = {}
        if type(rv) == str:
            rv = self.problem.net[rv]
        if (rv.name, val) not in self.rv_index_mem:
            self.rv_index_mem[(rv.name, val)] = rv.values.index(val)
        return self.rv_index_mem[(rv.name, val)]

    def theano_shared_weight(self, rng, n_in, n_out, indices, is_sigmoid = False):
        name = ''.join(map(str, indices))
        value = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6. / (n_in + n_out)),
                    high = np.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype = theano.config.floatX
            )
        if is_sigmoid:
            value *= 4
        w = theano.shared(value = value, name = 'w_' + name, borrow = True)
        b = theano.shared(
            value = np.zeros((n_out,), dtype=theano.config.floatX),
            name = 'b_' + name,
            borrow = True,
        )
        return w, b

    def init_model(self):
        rng = np.random.RandomState(1234)
        srng = RandomStreams(1234)
        ### activate function ###
        ### (X_t) => (sampled rv_ind for each rv, log sampling prob) ###
        hidden_frac = 2. / 3
        n_s = len(self.problem.net) - 1
        n_in = sum(len(self.problem.net[rv]) for rv in self.problem.rvs)
        weights = []
        output_distns = []
        output_cates = []
        output_inds = []
        output_prob = np.float64(1.)
        x = T.lvector('x')
        inputs = []
        for i, rv in enumerate(self.problem.rvs):
            n_val = len(self.problem.net[rv])
            inputs.append(T.extra_ops.to_one_hot(x[i:i+1], n_val))
        inputs = T.flatten(T.concatenate(inputs, axis = 1))
        cum_n_val = 0
        for i, rv in enumerate(self.problem.rvs):
            # size structure
            n_val = len(self.problem.net[rv])
            n_out = n_s + n_val
            n_hidden = int(hidden_frac * (n_in + n_out))
            # init weights
            w_0, b_0 = self.theano_shared_weight(rng, n_in, n_hidden, (i, 0))
            w_1, b_1 = self.theano_shared_weight(rng, n_hidden, n_out, (i, 1), True)
            # perform activation
            h = T.tanh(T.dot(inputs, w_0) + b_0)
            y = T.dot(h, w_1) + b_1
            # sample for this rv
            distn = T.nnet.softmax(y[:-n_s]) # has to be 2-D to be used below
            cate_chosen = T.flatten(srng.multinomial(n = 1, pvals = distn))
            ind_chosen = T.argmax(cate_chosen)
            # append necessary information
            weights.append((w_0, b_0, w_1, b_1))
            output_distns.append(distn)
            output_prob *= distn[0, ind_chosen]
            output_cates.append(cate_chosen)
            output_inds.append(ind_chosen)
            # prepare next rv's inputs
            s = T.nnet.sigmoid(y[-n_s:]) # secrets for next rounds!
            inputs = T.concatenate(output_cates + [s])
            cum_n_val += n_val
            n_in = cum_n_val + n_s
            n_s -= 1
        self.activate = theano.function([x], output_inds + [output_prob])
        ### update function (gradient ascent) ###
        ### A version that works on multiple rows and use given proposal (no sampling)
        mx = T.lmatrix('mx')
        mprop = T.lmatrix('mproposal')
        h = sjd = T.dvector('sjd')
        is_weights = is_denom = T.dvector('is_denom')
        bn_log_px = T.dvector('bn_log_px')
        bn_log_pprop = T.dvector('bn_log_pprop')
        x2p_log_p = T.zeros(sjd.shape)
        p2x_log_p = T.zeros(sjd.shape)
        arange = T.arange(is_denom.shape[0])
        x2p_inputs = []
        p2x_inputs = []
        for i, rv in enumerate(self.problem.rvs):
            n_val = len(self.problem.net[rv])
            x2p_inputs.append(T.extra_ops.to_one_hot(mx[:, i], n_val))
            p2x_inputs.append(T.extra_ops.to_one_hot(mprop[:, i], n_val))
        n_s = len(self.problem.net) - 1
        x2p_inputs = T.concatenate(x2p_inputs, axis = 1)
        p2x_inputs = T.concatenate(p2x_inputs, axis = 1)
        x2p_cates = []
        p2x_cates = []
        for i, (rv, (w_0, b_0, w_1, b_1)) in \
            enumerate(zip(self.problem.rvs, weights)):
            n_val = len(self.problem.net[rv])
            # perform activationS
            x2p_hid = T.tanh(x2p_inputs.dot(w_0) + b_0)
            p2x_hid = T.tanh(p2x_inputs.dot(w_0) + b_0)
            x2p_y = T.dot(x2p_hid, w_1) + b_1
            p2x_y = T.dot(p2x_hid, w_1) + b_1
            x2p_distn = T.nnet.softmax(x2p_y[arange, :-n_s])
            p2x_distn = T.nnet.softmax(p2x_y[arange, :-n_s])
            # update proposal probabilities and IS weights
            x2p_p = x2p_distn[arange, mprop[arange, i]]
            is_weights = is_weights * x2p_p
            x2p_log_p += T.log(x2p_p)
            p2x_log_p += T.log(p2x_distn[arange, mx[arange, i]])
            # prepare next rv's inputs
            x2p_s = T.nnet.sigmoid(x2p_y[arange, -n_s:])
            p2x_s = T.nnet.sigmoid(p2x_y[arange, -n_s:])
            x2p_cates.append(T.extra_ops.to_one_hot(mprop[arange, i], n_val))
            p2x_cates.append(T.extra_ops.to_one_hot(mx[arange, i], n_val))
            x2p_inputs = T.concatenate(x2p_cates + [x2p_s], axis = 1)
            p2x_inputs = T.concatenate(p2x_cates + [p2x_s], axis = 1)
            n_s -= 1
        h *= T.exp(bn_log_pprop + p2x_log_p - bn_log_px - x2p_log_p).clip(0., 1.)
        target = T.dot(h, is_weights) / T.sum(is_weights)
        log_target = T.log(target)
        self.log_target = theano.function(
            [mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom],
            [log_target]
        )
        flat_weights = list(chain(*weights))
        g_weights = [T.grad(log_target, w) for w in flat_weights]
        train_rate = T.dscalar('train_rate')
        self.update = theano.function(
            [mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom, train_rate],
            [log_target, train_rate],
            updates = [
                (w, w + train_rate * gw)
                for w, gw in zip(flat_weights, g_weights)
            ],
        )
        ### proposal probability function ###
        self.proposal_prob = theano.function(
            [mx, mprop], [T.log(is_weights)],
            givens = {
                is_denom: T.ones((mx.shape[0],), dtype = 'float64')
            },
        )
        print 'theano model initialized'

    def gen_static_train_data(self, b):
        x, p, pp = [], [], []
        for _ in xrange(b):
            ta, tb = self.random_tuple(), self.random_tuple()
            pa, pb = self.log_prob_tuple(ta), self.log_prob_tuple(tb)
            aa, ab = self.tuple_to_array(ta), self.tuple_to_array(tb)
            pab, pba = self.proposal_prob([aa, ab], [ab, aa])[0]
            jsd = (aa != ab).sum()
            if pa + pab < pb + pba:
                x.append(aa)
                p.append(ab)
                pp.append(jsd * exp(pa + pab))
            else:
                x.append(ab)
                p.append(aa)
                pp.append(jsd * exp(pb + pba))
        return np.array(x), np.array(p), np.array(pp)

    def estimate_esjd(self, n, T):
        s = 0
        ps = [self.init_particle() for _ in xrange(n)]
        for _ in xrange(T):
            ps_new = map(lambda p: self.update_particle(p, False), ps)
            for old, new in zip(ps, ps_new):
                s += (new[0] != old[0]).sum()
            ps = ps_new
        return s / float(n * T)

    # def static_train(self, n, b, alpha_fn):
    #     for it in xrange(n):
    #         alpha = alpha_fn(it)
    #         mx, proposals, esjds = self.gen_static_train_data(b)
    #         n = mx.shape[0]
    #         batch_size = self.train_batch_size
    #         print 'iteration', it, 'alpha', alpha,
    #         print 'pretraining', self.log_target(mx, proposals, esjds)[0],
    #         alpha = self.train_alpha_fn(it)
    #         self.update(mx, proposals, esjds, alpha)
    #         print 'posttraining', self.log_target(mx, proposals, esjds)[0],
    #         print 'estimated esjd', self.estimate_esjd(30, 30)
    #         self.data = [], [], []
    #     print 'training done'

    def init(self):
        self.init_model()
        self.data = [], [], [], [], [], []
        # self.static_train(1000, 1000, lambda t: 3.)

    def array_to_tuple(self, a):
        return tuple(self.problem.net[rv].values[i] for rv, i in zip(self.problem.rvs, a))

    def tuple_to_array(self, t):
        return np.array([self.get_rv_index(rv, i) for rv, i in zip(self.problem.rvs, t)])

    def log_prob_array(self, array):
        return self.log_prob_tuple(self.array_to_tuple(array))

    def particle_to_tuple(self, p):
        return self.array_to_tuple(p[0])

    def random_tuple(self):
        return tuple((np.random.choice(self.problem.net[rv].values) for rv in self.problem.rvs))

    def init_particle(self):
        array = np.array([np.random.randint(len(self.problem.net[rv])) for rv in self.problem.rvs])
        return array, self.log_prob_array(array)

    def update_iteration(self, it):
        if it > 0 and it % self.train_int == 0 and self.problem.rvs:
            mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom = \
                map(np.array, self.data)
            print 'esjd', self.estimate_esjd(50, 50)
            print 'train_steps', self.train_steps
            print 'pretraining', self.log_target(mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom)[0],
            for train_it in xrange(self.train_steps):
                alpha = self.train_alpha_fn(it, train_it)
                # for batch_ind in xrange(n_batch):
                self.update(mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom, alpha)
            print 'posttraining', self.log_target(mx, mprop, bn_log_px, bn_log_pprop, sjd, is_denom)[0],
            print 'esjd', self.estimate_esjd(50, 50)
            self.data = [], [], [], [], [], []
            print 'training done'

    def update_particle(self, particle, record_train_data = True):
        array, log_prob_x = particle
        outputs = self.activate(array)
        proposal = np.array(outputs[:-1])
        log_prob_proposal = self.log_prob_array(proposal)
        to_prob = outputs[-1]
        log_to_prob = log(to_prob)
        log_back_prob = self.proposal_prob([proposal], [array])[0][0]
        a = exp(log_prob_proposal + log_back_prob - log_prob_x - log_to_prob)
        if record_train_data:
            # add training data
            sjd = (array != proposal).sum()
            self.data[0].append(array)
            self.data[1].append(proposal)
            self.data[2].append(log_prob_x)
            self.data[3].append(log_prob_proposal)
            self.data[4].append(sjd)
            self.data[5].append(1 / to_prob)
        if np.random.uniform() < a:
            return proposal, log_prob_proposal
        else:
            return particle

