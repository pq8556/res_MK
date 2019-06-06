from nn_logging import getLogger
logger = getLogger()

# ______________________________________________________________________________
# Globals

#learning_rate = 0.00113
#learning_rate = 0.0033
learning_rate = 0.0063
print('learning_rate = ', learning_rate)
gradient_clip_norm = 100.

mask_value = 100.

discr_pt_cut = 8.

discr_pt_cut_high = 14.

reg_pt_scale = 100.
print('reg_pt_scale = ', reg_pt_scale)
discr_loss_weight = 20.
print('discr_loss_weight = ',discr_loss_weight)
add_noise = True

mixture = 6

l1_reg = 0.0

l2_reg = 0.0

dir1 = '/uscms_data/d3/mkim/MKWorkingArea/CMSSW_10_2_0/src/JL/Melrose/P2_CMSSW_10_1_5/src/L1TMuonSimulations/Analyzers/'

infile_muon = dir1 + 'test7/histos_tba.20.npz'

infile_pileup = dir1 + 'test7/histos_tbd.20.npz'

infile_muon_omtf = dir1 + 'test7/histos_tba_omtf.22.npz'

infile_pileup_omtf = dir1 + 'test7/histos_tbd_omtf.22.npz'


# ______________________________________________________________________________
# Import all the libs
import os
import sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
OLD_STDOUT = sys.stdout

logger.info('Using cmssw {0}'.format(os.environ['CMSSW_VERSION'] if 'CMSSW_VERSION' in os.environ else 'n/a'))

import numpy as np
np.random.seed(2023)
logger.info('Using numpy {0}'.format(np.__version__))

import tensorflow as tf
logger.info('Using tensorflow {0}'.format(tf.__version__))

import keras
from keras import backend as K
#K.set_epsilon(1e-08)
#K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4, allow_soft_placement=True)))
logger.info('Using keras {0}'.format(keras.__version__))
logger.info('.. list devices: {0}'.format(K.get_session().list_devices()))

import scipy
logger.info('Using scipy {0}'.format(scipy.__version__))

import sklearn
logger.info('Using sklearn {0}'.format(sklearn.__version__))

import matplotlib as mpl
import matplotlib.pyplot as plt
logger.info('Using matplotlib {0}'.format(mpl.__version__))
#%matplotlib inline

