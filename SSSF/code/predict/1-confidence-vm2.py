USAGE = """  
python 1-confidence.py <PART (dev|evl)> <MODEL>  
"""

import sys
if len(sys.argv) < 2:
    print USAGE
    sys.exit(1)

import os, os.path
import numpy
import cPickle
sys.path.append(os.path.expanduser('~/OpenSAT/SSSF/code/predict/RNN'))
from RNN import RNN
sys.path.append(os.path.expanduser("~/G/coconut"))
from fileutils import smart_open
from fileutils.htk import *
from scipy.io import savemat

# Paths  
INPUTFILE = sys.argv[1] # /vagrant/noisemes/
#PART = "evl.med"
#BASENAME = sys.argv[2]
INPUT_DIR = INPUTFILE+"/feature"
OUTPUT_DIR = INPUTFILE+"/hyp"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Load neural network
net = RNN(filename = os.path.expanduser('~/OpenSAT/SSSF/code/predict/model/noiseme.old/net.pkl.gz'))

# Load PCA matrix and scaling factors
with open(os.path.expanduser('~/OpenSAT/SSSF/code/predict/model/noiseme.old/pca.pkl'), 'rb') as f:
    locals().update(cPickle.load(f))
    with open(os.path.expanduser('~/OpenSAT/SSSF/code/predict/model/noiseme.old/scale.pkl'), 'rb') as f:
        w, b = cPickle.load(f)
        pca = lambda x: ((x[:,mask] - mu) / sigma).dot(V) * w + b

# Predict for each recording 
conf = {}
for filename in os.listdir(INPUT_DIR):
    print "Filename {}".format(filename)
    id,ext = os.path.splitext(filename)
    if ext != '.htk': continue
    print "Predicting for {} ...".format(id)
    feature = pca(readHtk(os.path.join(INPUT_DIR, filename))).astype('float32')
    x = feature.reshape((1,) + feature.shape)
    m = numpy.ones(x.shape[:-1], dtype = 'int32')
    conf[id] = net.predict(x, m)[0]

# Save predictions   
with smart_open(os.path.join(OUTPUT_DIR, 'confidence.pkl.gz'), 'wb') as f:
    cPickle.dump(conf, f)
    savemat(os.path.join(OUTPUT_DIR, 'confidence.mat'), conf)
