from ..structure import *
from ..utils import *

# Adaptive parent proposal
class AMCMC_BN(MCMC):
    def __init__(self, problem, verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        MCMC.__init__(self, problem, "AMCMC_BN", verbose_int, N, T, record_start)

    def particle_to_tuple(self, p):
        return p[0]

    def init_particle(self):
        t = tuple(np.random.choice(self.problem.net[rv].values) for rv in self.problem.rvs)
        return t, self.log_prob_tuple(t)

    def init(self):
        self.proposal = defaultdict(lambda: defaultdict(lambda: [{}, 0]))
        for rv in self.problem.rvs:
            obj = self.problem.net[rv]
            for k in obj.dict:
                for val in obj.values:
                    # Similar idea as Laplace's Rule of Succession
                    # each particle gives each value 0.5 fake 'visits'
                    v = obj.dict[k].pmf(val) * self.N / 2.0
                    self.proposal[rv][k][0][val] = v
                    self.proposal[rv][k][1] += v

    def sample(self, rv, key):
        total = self.proposal[rv][key][1]
        r = np.random.uniform(high = total)
        for k, v in self.proposal[rv][key][0].items():
            r -= v
            if r <= 0:
                return k, v / float(total)
        raise Exception("Unreached")

    def update_particle(self, particle):
        net = self.problem.net
        rvs = self.problem.rvs
        t, l = particle
        d = self.tuple_to_dict(t)
        rv = rvs[np.random.choice(len(rvs))]
        obj = net[rv]
        parent_t = obj._dict_to_tuple(d)
        np_rv = obj.dict[parent_t]
        sampled, prob = self.sample(rv, parent_t)
        new_d = d.copy()
        new_d[rv] = sampled
        log_a = -l - log(prob)
        new_l = l + np_rv.logpmf(sampled) - np_rv.logpmf(d[rv])
        for c in obj.children:
            new_l += net[c][new_d].logpmf(d[c]) - net[c][d].logpmf(d[c])
        log_a += new_l + log(self.proposal[rv][parent_t][0][d[rv]]) - log(self.proposal[rv][parent_t][1])
        a = min(1, exp(log_a))
        if self.bernoulli(a):
            for rv, val in new_d.items():
                if rv not in rvs:
                    continue
                key_t = self.problem.net[rv]._dict_to_tuple(new_d)
                self.proposal[rv][key_t][0][val] += 1
                self.proposal[rv][key_t][1] += 1
            return self.dict_to_tuple(new_d), new_l
        else:
            for rv, val in d.items():
                if rv not in rvs:
                    continue
                key_t = self.problem.net[rv]._dict_to_tuple(d)
                self.proposal[rv][key_t][0][val] += 1
                self.proposal[rv][key_t][1] += 1
            return particle
