from ..structure import *
from ..utils import *

class ParentProposal(MCMC):
    def __init__(self, problem, verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        MCMC.__init__(self, problem, "Parent Proposal", verbose_int, N, T, record_start)

    def particle_to_tuple(self, p):
        return p[0]

    def init_particle(self):
        t = tuple(random.choice(self.problem.net[rv].values) for rv in self.problem.rvs)
        return t, self.problem.net.log(self.tuple_to_dict(t))

    def update_particle(self, particle):
        net = self.problem.net
        rvs = self.problem.rvs
        t, l = particle
        d = self.tuple_to_dict(t)
        rv = rvs[np.random.choice(len(rvs))]
        obj = net[rv]
        parent_t = obj._dict_to_tuple(d)
        np_rv = obj.dict[parent_t]
        sampled = np_rv.rvs()
        new_d = d.copy()
        new_d[rv] = obj.number_to_value[sampled]
        log_a = -l - np_rv.logpmf(sampled)
        new_l = l + np_rv.logpmf(sampled) - np_rv.logpmf(d[rv])
        for c in obj.children:
            new_l += net[c][new_d].logpmf(d[c]) - net[c][d].logpmf(d[c])
        log_a += new_l + np_rv.logpmf(d[rv])
        a = min(1, exp(log_a))
        if np.random.uniform() < a:
            return self.dict_to_tuple(new_d), new_l
        else:
            return particle
