from amcmc.structure import *
from amcmc.inference import *
import numpy as np

np.random.seed(14361436)

T, F = True, False

bn = BayesNet()
b = CPT(bn, "b", [], (T, F))
b.add_entry({}, {T: 0.001, F: 0.999})
e = CPT(bn, "e", [], (T, F))
e.add_entry({}, {T: 0.002, F: 0.998})
a = CPT(bn, "a", ["e", "b"], (T, F))
a.add_entry({"b": T, "e": T}, {T: 0.95, F: 0.05})
a.add_entry({"b": T, "e": F}, {T: 0.94, F: 0.06})
a.add_entry({"b": F, "e": T}, {T: 0.29, F: 0.71})
a.add_entry({"b": F, "e": F}, {T: 0.001, F: 0.999})
j = CPT(bn, "j", ["a"], (T, F))
j.add_entry({"a": T}, {T: 0.9, F: 0.1})
j.add_entry({"a": F}, {T: 0.05, F: 0.95})
m = CPT(bn, "m", ["a"], (T, F))
m.add_entry({"a": T}, {T: 0.7, F: 0.3})
m.add_entry({"a": F}, {T: 0.01, F: 0.99})
r = CPT(bn, "r", ["m"], (T, F))
r.add_entry({"m": T}, {T: 0.9, F: 0.1})
r.add_entry({"m": F}, {T: 0.001, F: 0.999})
g = CPT(bn, "g", ["r", "e"], (T, F))
g.add_entry({"r": T, "e": T}, {T: 0.8, F: 0.2})
g.add_entry({"r": T, "e": F}, {T: 0.7, F: 0.3})
g.add_entry({"r": F, "e": T}, {T: 0.1, F: 0.9})
g.add_entry({"r": F, "e": F}, {T: 0.0001, F: 0.9999})
bn.finalize()

p = Problem(bn, {"j": F, "g": T}, ["b"])

# s = AMCMC_NN(p, verbose_int = 20, N = 200, T = 100, record_start = 5, \
#     train_int = 4, train_steps = 20, train_batch_size = 10, \
#     train_lambda_fn = lambda t: 1 / (10 + t ** 2),
#     train_alpha_fn = lambda t: 20.0 / (20 + t),
#     explore_ratio_fn = lambda t: 0.0)
# print s.infer()

# s = AMCMC_BN(p, verbose_int = 20, N = 200, T = 200, record_start = 20)
# print s.infer()

# s = ParentProposal(p, verbose_int = 20, N = 200, T = 200, record_start = 20)
# print s.infer()

# s = Gibbs(p, verbose_int = 100, N = 200, T = 1000, record_start = 200)
# print s.infer()

# s = BinaryBNOneVarMaxESJD(p, verbose_int = 100, N = 200, T = 1000, record_start = 200)
# print s.infer()

s = AMCMC_NN_ESJD(p, verbose_int = 50, N = 200, T = 200, record_start = 5,
    train_int = 20, train_steps = 1000, train_batch_size = 5,
    train_alpha_fn = lambda t, it: 10. / (10 + 0.1 * t + 0.02 * it))
print s.infer()

s = GibbsAll(p, verbose_int = 50, N = 200, T = 200, record_start = 5)
print s.infer()

# s = BinaryBNOneVarMaxESJDAll(p, verbose_int = 100, N = 200, T = 1000, record_start = 200)
# print s.infer()

s = PerfectProposal(p, verbose_int = 50, N = 200, T = 200, record_start = 5)
print s.infer()

