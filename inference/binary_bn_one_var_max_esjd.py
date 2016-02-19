from ..structure import *
from ..utils import *

# Assume each node in BN is binary (T, F)
# Considers all transitions w/ changes only related to one node, and thus used
# to compare with Gibbs.
# ESJD: expected square jump distance

class BinaryBNOneVarMaxESJD(MCMC):
    def __init__(self, problem, verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        MCMC.__init__(self, problem, "Max ESJD w/ One Node Proposal in Binary BN", verbose_int, N, T, record_start)
        self.node_values = set((True, False))
        self.propose_cache = {} # (rv, blanket tuple) -> (cur_val -> change_prob)

    def particle_to_tuple(self, p):
        return p

    def init(self):
        print 'Checking that BN is binary ...'
        for rv in self.problem.net.rvs:
            assert set(self.problem.net[rv].values) == self.node_values
        print 'Passed check'

    def init_particle(self):
        return tuple((random.choice(self.problem.net[rv].values) for rv in self.problem.rvs))

    # Return (cur_val -> change_prob)
    def get_prop_distn_for_rv_blanket(self, rv, d):
        blanket_t = self.problem.net[rv].dict_to_blanket_tuple(d)
        cache_key = (rv, blanket_t)
        if cache_key not in self.propose_cache:
            d = d.copy()
            d[rv] = True
            log_p_T = self.log_prob_dict(d)
            d[rv] = False
            log_p_F = self.log_prob_dict(d)
            if log_p_F < log_p_T:
                result = {False: 1.0, True: exp(log_p_F - log_p_T)}
            else:
                result = {True: 1.0, False: exp(log_p_T - log_p_F)}
            self.propose_cache[cache_key] = result
        return self.propose_cache[cache_key]

    def update_particle(self, particle):
        d = self.tuple_to_dict(particle)
        rv = self.problem.rvs[np.random.choice(len(self.problem.rvs))]
        prop_distn = self.get_prop_distn_for_rv_blanket(rv, d)
        if self.bernoulli(prop_distn[d[rv]]):
            d[rv] = not d[rv]
        return self.dict_to_tuple(d)
