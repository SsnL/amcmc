from mcmc import *
from itertools import product

raise Exception('Outdated')

random.seed(14361436)
np.random.seed(14361436)

vals = tuple(x for x in xrange(10))
uniform = {}
s = 0.0
for i, x in enumerate(vals):
    if i == len(vals) - 1:
        uniform[x] = 1 - s
    else:
        uniform[x] = 1.0 / len(vals)
        s += uniform[x]
result_vals = tuple(x for x in range(max(vals) + max(vals) + 1))
bn = BayesNet()
x = CPT(bn, 'x', [], vals)
x.add_entry({}, uniform)
y = CPT(bn, 'y', [], vals)
y.add_entry({}, uniform)
z = CPT(bn, 'z', ['x', 'y'], result_vals)
for x_v, y_v in product(vals, vals):
    z.add_entry({'x': x_v, 'y': y_v}, {z: float(z == x_v + y_v) for z in result_vals})
copy_z = CPT(bn, 'copy_z', ['z'], result_vals)
for z_v in result_vals:
    copy_z.add_entry({'z': z_v}, {z: float(z == z_v) for z in result_vals})
bn.finalize()

p = Problem(bn, {'x': 3, 'copy_z': 7}, ['y'])

s = AMCMC_NN(p, verbose_int = 20, N = 200, T = 200, train_int = 20, train_steps = 50, train_batch_size = 4, train_lambda = 0.0)
print s.infer()

s = Gibbs(p, verbose_int = 20, N = 200, T = 200)
print s.infer()

# s = PerfectProposal(p, verbose_int = 20, N = 200, T = 200)
# print s.infer()

# p = Problem(bn, {"j": T, "m": T}, ["b"])
# s = AMCMC_BN(p, verbose_int = 20, N = 200, T = 200)
# print s.infer()

# p = Problem(bn, {"j": T, "m": T}, ["b"])
# s = ParentProposal(p, verbose_int = 20, N = 200, T = 200)
# print s.infer()

# p = Problem(bn, {"b": T, "e": T}, ["j", "m"])
# print(p)
# s = Gibbs(p, N = 500, T = 1000)
# print s.infer()

