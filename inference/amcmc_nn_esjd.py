from ..structure import *
from ..utils import *
from neuron.neuralnet import NN

import numpy as np
import theano
import theano.tensor as T

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

class AMCMC_NN_ESJD(MCMC):
    def __init__(self, problem, \
            verbose_int = 100, N = 1000, T = 10000, record_start = 3000, \
            train_int = 100, train_steps = 1000, train_batch_size = 5, \
            train_lambda_fn = lambda t: 0.0,
            train_alpha_fn = lambda t: 600.0 / (600 + t)):
        MCMC.__init__(self, problem, "AMCMC_NN_ESJD", verbose_int, N, T, record_start)
        self.train_steps = train_steps
        self.train_int = train_int
        self.train_batch_size = train_batch_size
        self.train_lambda_fn = train_lambda_fn
        self.train_alpha_fn = train_alpha_fn

    def nn_input_segment(self, input_len, index):
        input_segment = [-1.0 for _ in xrange(input_len)]
        input_segment[index] = 1.0
        return input_segment

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

    def init_model(self):
        pass

    def init(self):
        d = {}
        n_in = 0
        for rv in self.problem.rvs:
            n_out = len(self.problem.net[rv])
            if n_in == 0:
                d[rv] = [1.0 / n_out for _ in xrange(n_out)]
            else:
                d[rv] = NN( \
                    [
                        n_in,
                        max(n_in * 4 // 3, n_out),
                        max(n_in, n_out),
                        n_out,
                    ], \
                    ['tanh', 'tanh', 'softmax'], \
                    cost_function = 'softmax_ce',
                )
            n_in += n_out
        self.nn_dict = d
        self.train_obs = []
        print self.get_nn_cpts_readble()

    def particle_to_tuple(self, p):
        return p[0]

    def init_particle(self):
        t = tuple(random.choice(self.problem.net[rv].values) for rv in self.problem.rvs)
        return t, self.nn_log_prob_tuple(t)

    # Sample RV according to PMF.
    # Return (index of sampled value in RV, sampled value, it's log probability)
    def sample(self, pmf, rv):
        total = sum(pmf)
        r = np.random.uniform(high = total)
        for (i, k), v in zip(enumerate(rv.values), pmf):
            r -= v
            if r <= 0:
                return i, k, log(v) - log(total)
        raise Exception("Unreached")

    # Sampled tuple, log probability that it is sampled
    def sample_nn(self):
        sampled = []
        nn_input = []
        l = 0
        for rv in self.problem.rvs:
            net = self.nn_dict[rv]
            rv_obj = self.problem.net[rv]
            if type(net) == list:
                pmf = net
            else:
                pmf = net.predict(nn_input)
            i, value, log_p = self.sample(pmf, rv_obj)
            add_input = self.nn_input_segment(len(rv_obj), i)
            nn_input.extend(add_input)
            sampled.append(value)
            l += log_p
        return tuple(sampled), l

    def nn_log_prob_tuple(self, t):
        nn_input = []
        l = 0
        for value, rv in zip(t, self.problem.rvs):
            net = self.nn_dict[rv]
            rv_obj = self.problem.net[rv]
            if type(net) == list:
                pmf = net
            else:
                pmf = net.predict(nn_input)
            i = self.get_rv_index(rv, value)
            add_input = self.nn_input_segment(len(rv_obj), i)
            nn_input.extend(add_input)
            l += log(pmf[i])
        return l

    def update_iteration(self, it):
        if it > 0 and it % self.train_int == 0 and self.problem.rvs:
            alpha = self.train_alpha_fn(it)
            lamda = self.train_lambda_fn(it)
            explore_ratio = self.explore_ratio_fn(it)
            print 'train at iteration {it} with alpha = {alpha}, lambda = {lamda}, {exp} random data'.format(it = it, alpha = alpha, lamda = lamda, exp = explore_ratio)
            n_obs = len(self.train_obs)
            n_exp = int(n_obs * explore_ratio)
            explore_obs_i = np.random.choice(n_obs, n_exp)
            # training
            inputs = [[] for _ in self.train_obs]
            rv = self.problem.rvs[0]
            rv_obj = self.problem.net[rv]
            # Laplace's rule of succession
            for i in xrange(len(self.nn_dict[rv])):
                self.nn_dict[rv][i] = 1
            # update the first rv's estimate table
            for obs_i, s in enumerate(self.train_obs):
                if obs_i in explore_obs_i:
                    rv_val_i = np.random.randint(len(rv_obj))
                else:
                    rv_val_i = self.get_rv_index(rv_obj, s[0])
                self.nn_dict[rv][rv_val_i] += 1
                add_input = self.nn_input_segment(len(rv_obj), rv_val_i)
                inputs[obs_i].extend(add_input)
            # train each of the rest rv's nn
            rv_i = 1
            for rv in self.problem.rvs[1:]:
                rv_obj = self.problem.net[rv]
                targets = []
                for obs_i, s in enumerate(self.train_obs):
                    if obs_i in explore_obs_i:
                        rv_val_i = np.random.randint(len(rv_obj))
                    else:
                        rv_val_i = self.get_rv_index(rv_obj, s[rv_i])
                    target = self.nn_input_segment(len(rv_obj), rv_val_i)
                    targets.append(target)
                print 'train', rv, '...'
                err = self.nn_dict[rv].train(inputs, targets, \
                    batch_size = self.train_batch_size, alpha = alpha, \
                    lamda = lamda, iterations = self.train_steps, \
                    calculate_errors = True)
                print 'avg error:', sum(err) / len(err)
                for obs_i, target in enumerate(targets):
                    inputs[obs_i].extend(target)
                rv_i += 1
            # clearing
            self.train_obs = []
            print self.get_nn_cpts_readble()

    def get_cpt_readable(self, rv, cpt, true_posterior, evidence_str):
        cpt_str = ''
        for givens, probs in cpt.items():
            given_str = ', '.join(['{grv}: {gval}'.format(grv = grv, gval = gval) for grv, gval in givens])
            if evidence_str:
                if given_str:
                    given_str += ', ' + evidence_str
                else:
                    given_str = evidence_str
            given_posterior_keys = filter(lambda t: all([t[i] == val for i, (rv, val) in enumerate(givens)]), true_posterior)
            given_total_posterior = sum(map(true_posterior.__getitem__, given_posterior_keys))
            for i, val in enumerate(self.problem.net[rv].values):
                val_given_posterior_keys = filter(lambda p: p[len(givens)] == val, given_posterior_keys)
                val_given_total_posterior = sum(map(true_posterior.__getitem__, val_given_posterior_keys))
                cpt_str += '\t({rv}: {val} | {given}) w.p. {prob}\tTrue posterior: {posterior}\n'.format( \
                    rv = rv, \
                    val = val, \
                    given = given_str, \
                    prob = probs[i], \
                    posterior = val_given_total_posterior / given_total_posterior, \
                )
        return '{rv}:\n{cpt}'.format(rv = rv, cpt = cpt_str)

    def get_nn_cpts_readble(self):
        if not self.problem.rvs:
            return ''
        evidence_str = self.get_evidence_readable()
        result = ''
        true_posterior = self.calc_exact_posterior()
        rv = self.problem.rvs[0]
        cpt = {}
        nn_inputs = {}
        rv_n_vals = len(self.problem.net[rv])
        s = float(sum(self.nn_dict[rv]))
        cpt[()] = map(lambda v: v / s, self.nn_dict[rv])
        for i, val in enumerate(self.problem.net[rv].values):
            nn_input = [0 for _ in xrange(rv_n_vals)]
            nn_input[i] = 1
            nn_inputs[((rv, val),)] = nn_input
        result += self.get_cpt_readable(rv, cpt, true_posterior, evidence_str)
        for rv in self.problem.rvs[1:]:
            cpt = {}
            new_nn_inputs = {}
            nn = self.nn_dict[rv]
            rv_n_vals = len(self.problem.net[rv])
            for readable, nn_input in nn_inputs.items():
                nn_outputs = nn.predict(nn_input)
                cpt[readable] = nn_outputs
                for (i, val), prob in zip(enumerate(self.problem.net[rv].values), nn_outputs):
                    new_nn_input = self.nn_input_segment(rv_n_vals, i)
                    # insert to left
                    new_nn_input[:0] = nn_input
                    new_nn_inputs[readable + ((rv, val),)] = new_nn_input
            nn_inputs = new_nn_inputs
            result += self.get_cpt_readable(rv, cpt, true_posterior, evidence_str)
        return result[:-1] # trim off the last '\n'

    def update_particle(self, particle):
        net = self.problem.net
        t, l = particle
        new_t, new_l = self.sample_nn()
        log_a = l + self.log_prob_tuple(new_t) - new_l - self.log_prob_tuple(t)
        a = min(1, exp(log_a))
        if self.bernoulli(a):
            result = new_t, new_l
        else:
            result = particle
        self.train_obs.append(result[0])
        return result
