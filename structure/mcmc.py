from scipy.stats import rv_discrete
from functools import reduce
from itertools import product
from collections import defaultdict, deque, OrderedDict
import matplotlib.pyplot as plt
from ..utils import *

class Problem:
    def __init__(self, net, evidence_dict, query_list):
        self.net = net
        # [query rvs ... hidden rvs]
        self.rvs = []
        non_query_rvs = []
        for rv in net.rvs:
            if rv in evidence_dict:
                continue
            elif rv in query_list:
                self.rvs.append(rv)
            else:
                non_query_rvs.append(rv)
        self.rvs.extend(non_query_rvs)
        self.evidence = evidence_dict
        self.query = query_list
        self.query_ind = tuple((self.rvs.index(q) for q in self.query))
        self._num_states = \
            reduce(mul, map(lambda rv: len(self.net[rv]), self.rvs))

    def __str__(self):
        return ", ".join(self.query) + " given " + \
            ", ".join(map(lambda p: "{0} = {1}".format(*p), self.evidence.items()))

    def __repr__(self):
        return str(self)

    ## Utility functions

    def get_num_states(self):
        return self._num_states

    # Ensures that D.items() has same order
    def ordered_comb_dict(self, rv_names):
        # defaultdict isn't used since order ought to be the same
        d = OrderedDict()
        for comb in product(*map(lambda rv: self.net[rv].values, rv_names)):
            d[comb] = 0
        return d

    @memoize_inst_meth
    def _tuple_to_dict(self, t):
        d = dict(zip(self.rvs, t))
        for e in self.evidence:
            d[e] = self.evidence[e]
        return d

    def tuple_to_dict(self, t):
        return self._tuple_to_dict(t).copy()

    @memoize_inst_meth
    def tuple_to_query(self, t):
        return tuple(t[i] for i in self.query_ind)

    @memoize_inst_meth
    def log_prob_tuple(self, t):
        return self.net.log(self.tuple_to_dict(t))

    @memoize_inst_meth
    def calc_exact_posterior(self):
        posterior = {}
        for comb in product(*map(lambda rv: self.net[rv].values, self.rvs)):
            posterior[comb] = exp(self.log_prob_tuple(comb))
        return normalize(posterior)

    @memoize_inst_meth
    def calc_exact_query_posterior(self):
        posterior = self.calc_exact_posterior()
        query_posterior = self.ordered_comb_dict(self.query)
        for t, prob in posterior.items():
            query_posterior[self.tuple_to_query(t)] += prob
        return query_posterior

    def prepare(self):
        self.net.problem = self

    def cleanup(self):
        self.net.clear_cache()

class MCMC:
    def __init__(self, problem, alg_name = "MCMC", verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        assert 0 <= record_start < T
        self.alg_name = alg_name
        self.verbose_int = verbose_int
        self.N = N
        self.T = T
        self.record_start = record_start
        self.problem = problem
        self.particles = None

    ## MCMC structure

    # Utils

    def tuple_to_dict(self, t):
        return self.problem.tuple_to_dict(t)

    def dict_to_tuple(self, d):
        return tuple(d[rv] for rv in self.problem.rvs)

    # Actual MCMC implementation might need extra/less info in each particle
    def particle_to_tuple(self, particle):
        raise NotImplementedError

    def init_particle(self):
        raise NotImplementedError

    def tuple_to_query(self, t):
        return self.problem.tuple_to_query(t)

    @memoize_inst_meth
    def particle_to_query(self, particle):
        return self.tuple_to_query(self.particle_to_tuple(particle))

    # Initialization

    # Init a wrapped particle
    def init_particle_wrapper(self):
        p = self.init_particle()
        if self.calc_eff:
            d = self.problem.ordered_comb_dict(self.problem.rvs)
        else:
            d = None
        self.record_particle(p, d = d)
        return p, d

    # Customized init function, run before the MCMC algorithm
    def init(self):
        pass

    def init_wrapper(self, calc_eff, plot_lag):
        self.init()
        self.t = 0
        self.calc_eff = calc_eff
        self.plot_lag = plot_lag
        if plot_lag != None:
            # These are dict query -> counts
            if plot_lag >= 0:
                self.plot_deque = deque()
                self.plot_deque.append( \
                    self.problem.ordered_comb_dict(self.problem.query))
            self.plot_summary = \
                self.problem.ordered_comb_dict(self.problem.query)
            self.plot_data = [], []
            self.plot_ref = np.array( \
                self.problem.calc_exact_query_posterior().values())
            plt.axes(xlim = (plot_lag, self.T), ylim = (0, 1))
            self.plot_line, = plt.plot(*self.plot_data)
            plt.ion()
            plt.show()
        self.counts = self.problem.ordered_comb_dict(self.problem.query)
        self.particles = [self.init_particle_wrapper() for _ in xrange(self.N)]

    def record_particle(self, p, t = None, q = None, d = None):
        if self.t >= self.record_start or self.plot_lag != None:
            if q == None:
                if t == None:
                    t = self.particle_to_tuple(p)
                q = self.tuple_to_query(t)
            if self.t >= self.record_start:
                self.counts[q] += 1
            if self.plot_lag != None:
                if self.plot_lag >= 0:
                    self.plot_deque[-1][q] += 1
                self.plot_summary[q] += 1
        if self.calc_eff:
            if t == None:
                t = self.particle_to_tuple(p)
            d[t] += 1
        return p

    # Update

    def update_particle(self, particle):
        raise NotImplementedError

    # Update a wrapped particle
    def update_particle_wrapper(self, particle_wrapped):
        old_p, d = particle_wrapped
        new_p = self.update_particle(old_p)
        self.record_particle(new_p, d = d)
        return new_p, d

    ## Customized function that run at the end of each iteration numbered IT
    def update_iteration(self, it):
        pass

    def update_wrapper(self):
        self.t += 1
        if self.t % self.verbose_int == 0:
            print "iteration", self.t
        if self.plot_lag >= 0:
            self.plot_deque.append( \
                self.problem.ordered_comb_dict(self.problem.query))
        self.particles = map(self.update_particle_wrapper, self.particles)
        if self.plot_lag != None:
            if self.plot_lag >= 0 and len(self.plot_deque) > self.plot_lag:
                out = self.plot_deque.popleft()
                for q, n in out.items():
                    self.plot_summary[q] -= 1
            if self.plot_lag < 0 or len(self.plot_deque) == self.plot_lag:
                diff = np.array(normalize(self.plot_summary).values()) - self.plot_ref
                self.plot_data[0].append(self.t)
                self.plot_data[1].append(diff.dot(diff))
                self.plot_line.set_xdata(self.plot_data[0])
                self.plot_line.set_ydata(self.plot_data[1])
                plt.ylim((0, max(self.plot_data[1])))
                plt.draw()
                plt.pause(0.0001)
        self.update_iteration(self.t)

    # Printing utils

    def print_params(self):
        print "N = {0}, T = {1}".format(self.N, self.T)

    def additional_args_str(self):
        return ''

    # Running the algorithm

    # The criteria is Var((\sum_{t = 0}^{T} X_t) / T)^{-1}, each particle
    # (p, d)'s trajectory is stored in its second entry (d).
    def _run(self, calc_eff, plot_lag):
        print "Algorithm: {alg}\nProblem:\n{prob}".format( \
            alg = self.alg_name, prob = self.problem)
        additional_args_str = self.additional_args_str()
        if additional_args_str:
            print additional_args_str
        self.init_wrapper(calc_eff, plot_lag)
        print "iteration", self.t
        while self.t <= self.T:
            self.update_wrapper()
        readable_res = self.get_result_readable(normalize(self.counts))
        if self.calc_eff:
            # The 2-D array to calculate the criteria, each row is the summary of a
            # particle.
            s = np.zeros((self.N, self.problem.get_num_states()))
            # Assume D has the same order
            for i, (p, d) in enumerate(self.particles):
                s[i] = np.array(normalize(d).values())
            mean = s.mean(axis = 0)
            eff = (self.N - 1.0) / ((s - mean) ** 2).sum()
            return "Result:\n{res}\nempirical efficiency: {eff}".format( \
                res = readable_res, eff = eff)
        else:
            return "Result:\n{res}".format(res = readable_res)

    # plot_lag = -1 means all
    def infer(self, calc_eff = True, plot_lag = None):
        self.problem.prepare()
        result = self._run(calc_eff, plot_lag)
        self.problem.cleanup()
        return result

    ## MH utility functions

    def log_prob_tuple(self, t):
        return self.problem.log_prob_tuple(t)

    def log_prob_dict(self, d):
        return self.problem.net.log(d)

    def mh_acc_log_prob(self, cur_tuple, tar_tuple, log_to_prob, log_back_prob):
        return self.log_prob_tuple(tar_tuple) + log_back_prob - \
            self.log_prob_tuple(cur_tuple) - log_to_prob

    def bernoulli(self, p):
        return np.random.uniform() <= p

    ## Output functions

    def get_evidence_readable(self):
        return ', '.join( \
            '{0}: {1}'.format(*p) for p in self.problem.evidence.items() \
        )

    def get_result_readable(self, result):
        entries = []
        evidence_str = self.get_evidence_readable()
        query_rv_to_rvs_ind = {}
        query_posterior = self.problem.calc_exact_query_posterior()
        for rv in self.problem.query:
            query_rv_to_rvs_ind[rv] = self.problem.rvs.index(rv)
        for query, prob in result.items():
            query_str = ", ".join( \
                "{0}: {1}".format(*p) for p in zip(self.problem.query, query) \
            )
            if evidence_str:
                query_str += ' | ' + evidence_str
            entries.append( \
                '\t({query}) w.p. {prob}\tTrue posterior: {posterior}'.format( \
                query = query_str, \
                prob = prob, \
                posterior = query_posterior[query], \
            ))
        return "\n".join(entries)
