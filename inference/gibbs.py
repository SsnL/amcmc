from ..structure import *
from ..utils import *

class Gibbs(MCMC):
    def __init__(self, problem, verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        MCMC.__init__(self, problem, "Gibbs", verbose_int, N, T, record_start)

    def particle_to_tuple(self, p):
        return p

    def init_particle(self):
        return tuple((np.random.choice(self.problem.net[rv].values) for rv in self.problem.rvs))

    def update_particle(self, particle):
        d = self.tuple_to_dict(particle)
        rv = self.problem.rvs[np.random.choice(len(self.problem.rvs))]
        d[rv] = self.problem.net[rv].sample_blanket(d)
        return self.dict_to_tuple(d)

