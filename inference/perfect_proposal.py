from ..structure import *
from ..utils import *

class PerfectProposal(MCMC):
    def __init__(self, problem, verbose_int = 100, N = 1000, T = 10000, record_start = 3000):
        MCMC.__init__(self, problem, "Perfect Proposal", verbose_int, N, T, record_start)

    def particle_to_tuple(self, p):
        return p

    def init(self):
        self.perfect_proposal = self.problem.calc_exact_posterior()

    def perfect_sample(self):
        r = np.random.uniform()
        for k, v in self.perfect_proposal.items():
            r -= v
            if r <= 0:
                return k
        return k

    def init_particle(self):
        return self.perfect_sample()

    def update_particle(self, particle):
        return self.perfect_sample()
