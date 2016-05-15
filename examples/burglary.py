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
# r = CPT(bn, "r", ["m"])
# r.add_entry({"m": T}, {T: 0.9, F: 0.1})
# r.add_entry({"m": F}, {T: 0.001, F: 0.999})
# g = CPT(bn, "g", ["r"])
# g.add_entry({"r": T}, {T: 0.7, F: 0.3})
# g.add_entry({"r": F}, {T: 0.001, F: 0.999})
bn.finalize()

p = Problem(bn, {"j": T, "m": T}, ["b"])

# s = AMCMC_NN(p, verbose_int = 20, N = 200, T = 50, record_start = 10,
#     train_int = 2, train_steps = 10,
#     train_batch_size = 10)
# print s.infer()

# s = AMCMC_BN(p, verbose_int = 20, N = 200, T = 200, record_start = 20)
# print s.infer()

# s = ParentProposal(p, verbose_int = 20, N = 200, T = 200, record_start = 20)
# print s.infer()

# s = Gibbs(p, verbose_int = 20, N = 200, T = 50, record_start = 10)
# print s.infer()

# s = BinaryBNOneVarMaxESJD(p, verbose_int = 20, N = 200, T = 500, record_start = 10)
# print s.infer(False)

s = AMCMC_NN_ESJD(p, verbose_int = 20, N = 200, T = 860, record_start = 270,
    train_int = 20, train_steps = 1000, train_batch_size = 5,
    train_alpha_fn = lambda t, it: 10. / (-30 + 2 * t + 0.05 * it),
    train_stop_it = 260)
print s.infer()

s = GibbsAll(p, verbose_int = 20, N = 200, T = 600, record_start = 10)
print s.infer()

s = BinaryBNOneVarMaxESJDAll(p, verbose_int = 20, N = 200, T = 600, record_start = 10)
print s.infer()

# s = PerfectProposal(p, verbose_int = 20, N = 200, T = 10, record_start = 5)
# print s.infer(False)

