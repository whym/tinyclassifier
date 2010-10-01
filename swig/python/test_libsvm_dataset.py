from TinyClassifier import *
import unittest
import urllib
import tempfile
import os
import re
import StringIO
from collections import namedtuple
import random
import timeit
import sys

space_delimiter = re.compile('\s+')

def parse(lines):
    global space_delimiter
    for line in lines:
        line = line.strip().split('#')[0]
        if len(line) == 0:
            continue
        cols = space_delimiter.split(line)
        yield (int(cols[0]), [(int(x[0]),float(x[1]))
                              for x in filter(lambda x: len(x) == 2,
                                              [tuple(x.split(':')[0:2]) for x in cols[1:]])])

def find_dim(lines):
    dim = 0
    for (_, vec) in parse(lines):
        for (i,v) in vec:
            if i >= dim:
                dim = i + 1
    return dim

def libsvm_to_vec_iter(lines):
    dim = find_dim(lines)
    for (lab, ls) in parse(lines):
        vec = FloatVector(dim)
        for (i,v) in ls:
            if i < dim:
                vec[i] = v
        yield (lab, vec)

def libsvm_to_vec(lines):
    labs = IntVector()
    vecs = FloatVectorVector()
    for (lab,vec) in libsvm_to_vec_iter(lines):
        labs.push_back(lab)
        vecs.push_back(vec)
    return (labs, vecs)

def cached_urlopen(url):
    path = os.path.sep.join([tempfile.gettempdir(), urllib.quote(url, safe='').replace('%', '_')])
    if os.path.exists(path):
        return open(path)
    else:
        str = ''.join(list(urllib.urlopen(url)))
        open(path, 'w').write(str)
        return StringIO.StringIO(str)

class TestLibSVM(unittest.TestCase):
        
    def _test_libsvm(self, perp, data_train, data_test):
        pn_tuple = namedtuple('pn', 'p n')
        pn = pn_tuple({True: 0, False: 0},
                      {True: 0, False: 0})
        
        random.shuffle(data_train)
        labs, vecs = libsvm_to_vec(data_train)
        itr = perp.train(vecs, labs)
        for (lab,vec) in libsvm_to_vec_iter(data_test):
            pred = perp.predict(vec)
            correct = pred * lab > 0
            if pred > 0:
                pn.p[correct] += 1
            else:
                pn.n[correct] += 1
        print pn
        print ' dim = %d' % len(vecs[0])
        print ' iterations = %d' % itr
        print ' accuracy  = %f' % (float(pn.p[True] + pn.n[True]) / sum(pn.p.values() + pn.n.values()))
        prec = float(pn.p[True]) / sum(pn.p.values())
        reca = float(pn.p[True]) / (pn.p[True] + pn.n[False])
        print ' precision = %f' % prec
        print ' recall    = %f' % reca
        print ' fmeasure  = %f' % (1.0 / (0.5/prec + 0.5/reca))
        
    def test_libsvm1(self):
        dim = 122
        data_train, data_test = [list(cached_urlopen(x)) for x in
                                 ['http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a',
                                  'http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t']]
        for (name,x) in [('linear',    FloatPerceptron(dim,   4)),
                         ('1st-order', FloatPKProjectron(dim, 4, 1, 0, 0)),
                         ('3rd-order', FloatPKPerceptron(dim, 4, 3, 1, 0))]:
            random.seed(1029)
            t = timeit.Timer(lambda: self._test_libsvm(x, data_train, data_test))
            try:
                t.timeit(1)
            except:
                t.print_exc(sys.stderr)

if __name__ == '__main__':
    unittest.main()
