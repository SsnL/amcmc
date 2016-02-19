from scipy.stats import rv_discrete
from functools import reduce
from collections import defaultdict
from ..utils import *

# number-valued rvs
class CPT:
    def __init__(self, net, name, parents, values):
        self.net = net
        self.name = str(name)
        self.parents = tuple(sorted(parents))
        self.values = set(values)
        if len(self.parents) != len(parents):
            raise Exception("duplicated parent names")
        self.children = set()
        self.p_of_c = set()
        self.dict = {}
        self._final = False
        self._frozen = False
        self.blanket_cache = {}
        self.net.add_rv(self.name, self)
        self._problem = None
        self.number_to_value = {}
        for v in values:
            self.number_to_value[int(v)] = v

    @property
    def problem(self):
        if not self._problem:
            raise Exception("set problem first")
        return self._problem

    @problem.setter
    def problem(self, value):
        if value != self.net.problem:
            raise Exception("set problem field on net")
        if value != self._problem:
            self.clear_cache()
        self._problem = value

    def __len__(self):
        self.check_frozen()
        return len(self.values)

    @property
    def final(self):
        return self._final

    @property
    def frozen(self):
        return self._frozen

    def _dict_to_tuple(self, dict_k):
        return tuple(dict_k[k] for k in self.parents)

    def add_entry(self, parent_dict, prob_dict):
        self.check_editable()
        assert type(parent_dict) == type(prob_dict) == dict, \
            'both arguments must be dictionaries'
        tuple_key = self._dict_to_tuple(parent_dict)
        if sum(prob_dict.values()) != 1.0 or len(tuple_key) != len(self.parents):
            raise Exception("invalid entry")
        for v in prob_dict.keys():
            if v not in self.values:
                raise Exception('{0} invalid value of {1}'.format(v, self.name))
        self.dict[tuple_key] = \
            rv_discrete(name = self.name, \
                values = (tuple(prob_dict.keys()), tuple(prob_dict.values())))

    def check_final(self):
        if not self.final:
            raise Exception("unfinalized CPT")

    def check_editable(self):
        if self.frozen:
            raise Exception("already frozen CPT")

    def check_frozen(self):
        if not self.frozen:
            raise Exception("freeze this before finalize")

    def sample(self, parent_dict):
        self.check_final()
        return self.number_to_value[self[parent_dict].rvs()]

    def __getitem__(self, parent_dict):
        self.check_final()
        return self.dict[self._dict_to_tuple(parent_dict)]

    def __str__(self):
        s = ''
        if not self.final:
            s += 'WARNING: {name} is not final'.format(name = self.name)
        # for parent_dict,

    def clear_cache(self):
        self.blanket_cache = {}

    def freeze(self):
        self.check_editable()
        self._frozen = True

    # only checkes that numbers match, not entry keys
    def finalize_a(self):
        self.check_frozen()
        if len(self.dict) != \
            reduce(mul, map(lambda p: len(self.net[p]), self.parents), 1):
            print self.dict, self.parents
            raise Exception("{0} expects more entries".format(self.name))
        for p in self.parents:
            self.net[p].children.add(self.name)
            for other_p in self.parents:
                if p != other_p:
                    self.net[p].p_of_c.add(other_p)

    def finalize_b(self):
        self.children = tuple(self.children)
        self.p_of_c = tuple(self.p_of_c)
        self.values = tuple(self.values)
        self._final = True

    def dict_to_blanket_tuple(self, d):
        l = []
        for p in self.parents:
            l.append(d[p])
        for c in self.children:
            l.append(d[c])
        for o in self.p_of_c:
            l.append(d[o])
        return tuple(l)

    def rv_blanket(self, d):
        self.check_final()
        tuple_key = self.dict_to_blanket_tuple(d)
        if tuple_key not in self.blanket_cache:
            result = defaultdict(float)
            def add_entry(v):
                d[self.name] = v
                result[v] += self[d].logpmf(v)
                for c in self.children:
                    result[v] += self.net[c][d].logpmf(d[c])
            for v in self.values:
                add_entry(v)
            log_normalize(result)
            self.blanket_cache[tuple_key] = \
                rv_discrete(name = self.name, \
                    values = (tuple(result.keys()), tuple(map(exp, result.values()))))
        return self.blanket_cache[tuple_key]

    def sample_blanket(self, d):
        self.check_final()
        return self.number_to_value[self.rv_blanket(d).rvs()]

class BayesNet:
    def __init__(self):
        self.rvs = {}
        self.unresolved_rvs = set()
        self.ordered_rvs = None
        self._final = False
        self._problem = None

    @property
    def problem(self):
        if not self._problem:
            raise Exception("set problem first")
        return self._problem

    @problem.setter
    def problem(self, value):
        if value != self._problem:
            self.clear_cache()
        self._problem = value
        for rv in self.rvs.values():
            rv.problem = value

    def clear_cache(self):
        for rv in self.rvs.values():
            rv.clear_cache()

    @property
    def final(self):
        return self._final

    def __getitem__(self, rv):
        return self.get_cpt(rv)

    def get_cpt(self, rv):
        if rv not in self.rvs:
            raise Exception("rv not found: {0}".format(rv))
        return self.rvs[rv]

    def add_rv(self, rv, cpt):
        if self.final:
            raise Exception("cannot edit finalized net")
        for parent_rv in cpt.parents:
            if parent_rv not in self.rvs:
                self.unresolved_rvs.add(parent_rv)
        self.unresolved_rvs.discard(rv)
        self.rvs[rv] = cpt

    def log(self, d):
        return sum(v[d].logpmf(d[k]) for k, v in self.rvs.items())

    def finalize(self):
        if self.unresolved_rvs:
            raise Exception("unreolved rvs: {0}".format(str(self.unresolved_rvs)))
        for rv in self.rvs:
            self.rvs[rv].freeze()
        for rv in self.rvs:
            self.rvs[rv].finalize_a()
        for rv in self.rvs:
            self.rvs[rv].finalize_b()
        visited = set()
        ordered = []
        path = set()
        def explore(rv):
            path.add(rv)
            visited.add(rv)
            for parent_rv in self[rv].parents:
                if parent_rv in path:
                    raise Exception("cycle detected")
                if parent_rv not in visited:
                    explore(parent_rv)
            ordered.append(rv)
            path.remove(rv)
        for rv in self.rvs:
            explore(rv)
        self.ordered_rvs = ordered
        self._final = True
