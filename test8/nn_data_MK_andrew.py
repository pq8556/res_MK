import numpy as np
from MK_proc import *
#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nn_logging import getLogger
logger = getLogger()

from nn_encode import Encoder_MK, pt_range_select


# ______________________________________________________________________________
def muon_data(filename, adjust_scale=0, reg_pt_scale=1.0, correct_for_eta=False, nbits=7, MKptmin=20, MKptmax=40, get_eta=False, verbose=False):
    try:
        logger.info('Loading muon data from {0} ...'.format(filename))
        loaded = np.load(filename)
        the_variables_0, nanStd = det_bits(loaded,ind_det=24,nbits=nbits,verbose=verbose)
        the_parameters_0 = loaded['parameters']
        ooptmin = 1/float(MKptmin) # e.g. 0.05 (20GeV)
        ooptmax = 1/float(MKptmax) # e.g. 0.025 (40GeV)
        pttemp = pt_range_select(the_parameters_0[:,0],ptmin=ooptmax,ptmax=ooptmin)
        if verbose == True:
            print("hello")
        #print(pttemp[:10000])
        the_variables = the_variables_0[pttemp]
    
        the_parameters = the_parameters_0[pttemp]
        arrhead = the_parameters[:100,0]
        if verbose == True:
            print(arrhead)
            print(np.reciprocal(arrhead))
        logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
        logger.info('Loaded the parameters with shape {0}'.format(the_parameters.shape))
    except:
        logger.error('Failed to load data from file: {0}'.format(filename))
  

    print(the_variables.shape[0])
    assert(the_variables.shape[0] == the_parameters.shape[0])
  
    encoder = Encoder_MK(the_variables, the_parameters, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, drop_MK=True, nbits=nbits, nanStd=nanStd)
    if correct_for_eta:
        x, y, w, x_mask = encoder.get_x(), encoder.get_y_corrected_for_eta(), encoder.get_w(), encoder.get_x_mask()
    else:
        x, y, w, x_mask = encoder.get_x(), encoder.get_y(), encoder.get_w(), encoder.get_x_mask()
    logger.info('Loaded the encoded variables with shape {0}'.format(x.shape))
    logger.info('Loaded the encoded parameters with shape {0}'.format(y.shape))
    assert(np.isfinite(x).all())
    if get_eta == True:
        y_eta = encoder.get_y_eta()
        return x, y, y_eta, w, x_mask
    else:
        return x, y, w, x_mask


def muon_data_split(filename, adjust_scale=0, reg_pt_scale=1.0, test_size=0.5, correct_for_eta=False,nbits=7,\
                   MKptmin=2.000001, MKptmax=20000, get_eta=False):
    if get_eta == True:
        x, y, y_eta, w, x_mask = muon_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, correct_for_eta=correct_for_eta,nbits=nbits,MKptmin=MKptmin,MKptmax=MKptmax, get_eta=get_eta)
        # Split dataset in training and testing
        x_train, x_test, y_train, y_test, y_eta_train, y_eta_test, w_train, w_test, x_mask_train, x_mask_test = train_test_split(x, y, y_eta, w, x_mask, test_size=test_size)
    else:
        x, y, w, x_mask = muon_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, correct_for_eta=correct_for_eta,nbits=nbits,MKptmin=MKptmin,MKptmax=MKptmax, get_eta=get_eta)
        # Split dataset in training and testing
        x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = train_test_split(x, y, w, x_mask, test_size=test_size)
    

    
    logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
  
    # Check for cases where the number of events in the last batch could be too few
    validation_split = 0.1
    train_num_samples = int(x_train.shape[0] * (1.0-validation_split))
    val_num_samples = x_train.shape[0] - train_num_samples
    batch_size = 128
    if (train_num_samples%batch_size) < 100:
        logger.warning('The last batch for training could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
        logger.warning('Try this formula: int(int({0}*{1})*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
    train_num_samples = int(x_train.shape[0] * 2 * (1.0-validation_split))
    val_num_samples = x_train.shape[0] - train_num_samples
    batch_size = 128
    if (train_num_samples%batch_size) < 100:
        logger.warning('The last batch for training after mixing could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
        logger.warning('Try this formula: int(int({0}*{1})*2*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
    if get_eta == True:
        return x_train, x_test, y_train, y_test, y_eta_train, y_eta_test, w_train, w_test, x_mask_train, x_mask_test
    else:
        return x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test


# ______________________________________________________________________________
def pileup_data(filename, adjust_scale=0, reg_pt_scale=1.0,nbits=7):
    try:
        logger.info('Loading pileup data from {0} ...'.format(filename))
        loaded = np.load(filename)
        the_variables, nanStd = det_bits(loaded,ind_det=24,nbits=nbits,verbose=True)
        #the_variables = loaded['variables']
        the_parameters = np.zeros((the_variables.shape[0], 3), dtype=np.float32)
        the_aux = loaded['aux']
        logger.info('Loaded the variables with shape {0}'.format(the_variables.shape))
        logger.info('Loaded the auxiliary PU info with shape {0}'.format(the_aux.shape))
    except:
        logger.error('Failed to load data from file: {0}'.format(filename))
  
    assert(the_variables.shape[0] == the_aux.shape[0])
    assert(the_aux.shape[1] == 4)  # jobid, ievt, highest_part_pt, highest_track_pt
  
    encoder = Encoder_MK(the_variables, the_parameters, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale, drop_MK=True, nbits=nbits, nanStd=nanStd)
    x, y, w, x_mask = encoder.get_x(), encoder.get_y(), encoder.get_w(), encoder.get_x_mask()
    logger.info('Loaded the encoded variables with shape {0}'.format(x.shape))
    logger.info('Loaded the encoded auxiliary PU info with shape {0}'.format(the_aux.shape))
    assert(np.isfinite(x).all())
    return x, the_aux, w, x_mask


def pileup_data_split(filename, adjust_scale=0, reg_pt_scale=1.0, test_job=50,nbits=7):
    x, aux, w, x_mask = pileup_data(filename, adjust_scale=adjust_scale, reg_pt_scale=reg_pt_scale,nbits=nbits)
  
    # Split dataset in training and testing
    split = aux[:,0].astype(np.int32) < test_job
    x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test = x[split], x[~split], aux[split], aux[~split], w[split], w[~split], x_mask[split], x_mask[~split]
    logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
    return x_train, x_test, aux_train, aux_test, w_train, w_test, x_mask_train, x_mask_test




# ______________________________________________________________________________
def mix_training_inputs(x_train, y_train, pu_x_train, pu_y_train, pu_aux_train, discr_pt_cut=14., tile=15):

  # Apply veto on PU events with a muon with pT > 14 GeV
  pu_x_train_tmp = ~(pu_aux_train[:,2] > discr_pt_cut)
  pu_x_train = pu_x_train[pu_x_train_tmp]
  pu_y_train = [pu_y_train[0][pu_x_train_tmp], pu_y_train[1][pu_x_train_tmp]]

  # Put together x_train & pu_x_train, y_train & pu_y_train
  assert(len(pu_y_train) == 2)
  assert(pu_x_train.shape[0] == pu_y_train[0].shape[0])
  assert(pu_x_train.shape[0] == pu_y_train[1].shape[0])
  num_samples = pu_x_train.shape[0]
  index_array = np.arange(num_samples)
  index_array_ext = np.tile(index_array, tile)  # choose tile to make sure pu_x_train_ext has more entries than x_train
  pu_x_train_ext = pu_x_train[index_array_ext]
  pu_y_train_ext = [pu_y_train[0][index_array_ext], pu_y_train[1][index_array_ext]]

  assert(len(y_train) == 2)
  assert(x_train.shape[0] == y_train[0].shape[0])
  assert(x_train.shape[0] == y_train[1].shape[0])
  if not (pu_x_train_ext.shape[0] >= x_train.shape[0]):
    raise Exception('pu_x_train_ext is required to have more entries than x_train. Make sure {0} >= {1}'.format(pu_x_train_ext.shape[0], x_train.shape[0]))
  num_samples = x_train.shape[0]
  index_array = np.arange(num_samples)
  #np.random.shuffle(index_array)

  try:
    from keras.engine.training import _make_batches as make_batches
  except ImportError:
    from keras.engine.training_utils import make_batches

  sample_batch_size = 128
  batches = make_batches(num_samples, sample_batch_size)

  x_train_new = np.zeros((num_samples*2, x_train.shape[1]), dtype=np.float32)
  y_train_new = [np.zeros((num_samples*2,), dtype=np.float32), np.zeros((num_samples*2,), dtype=np.float32)]

  for batch_index, (batch_start, batch_end) in enumerate(batches):
    batch_ids = index_array[batch_start:batch_end]
    x_train_new[batch_start*2:batch_start*2 + (batch_end-batch_start)] = x_train[batch_ids]
    x_train_new[batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_x_train_ext[batch_ids]
    y_train_new[0][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[0][batch_ids]
    y_train_new[0][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[0][batch_ids]
    y_train_new[1][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[1][batch_ids]
    y_train_new[1][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[1][batch_ids]

  logger.info('Mixed muon data with pileup data. x_train_new has shape {0}, y_train_new has shape {1},{2}'.format(x_train_new.shape, y_train_new[0].shape, y_train_new[1].shape))
  return x_train_new, y_train_new



# ______________________________________________________________________________
def mix_training_inputs_MK(x_train, y_train, pu_x_train, pu_y_train, pu_aux_train, discr_pt_cut=14., tile=15):
    # Apply veto on PU events with a muon with pT > 14 GeV
    pu_x_train_tmp = ~(pu_aux_train[:,2] > discr_pt_cut)
    pu_x_train = pu_x_train[pu_x_train_tmp]
    #MK: adding charge
    pu_y_train = [pu_y_train[0][pu_x_train_tmp], pu_y_train[1][pu_x_train_tmp], pu_y_train[2][pu_x_train_tmp]]
  
    # Put together x_train & pu_x_train, y_train & pu_y_train
    # MK: 3 components
    assert(len(pu_y_train) == 3)
    assert(pu_x_train.shape[0] == pu_y_train[0].shape[0])
    assert(pu_x_train.shape[0] == pu_y_train[1].shape[0])
    num_samples = pu_x_train.shape[0]
    index_array = np.arange(num_samples)
    # ext for extention
    index_array_ext = np.tile(index_array, tile)  # choose tile to make sure pu_x_train_ext has more entries than x_train
    pu_x_train_ext = pu_x_train[index_array_ext]
    #MK: adding charge
    pu_y_train_ext = [pu_y_train[0][index_array_ext], pu_y_train[1][index_array_ext], pu_y_train[2][index_array_ext]]
  
    # MK: 3 components
    assert(len(y_train) == 3)
    assert(x_train.shape[0] == y_train[0].shape[0])
    assert(x_train.shape[0] == y_train[1].shape[0])
    if not (pu_x_train_ext.shape[0] >= x_train.shape[0]):
        raise Exception('pu_x_train_ext is required to have more entries than x_train. Make sure {0} >= {1}'.format(pu_x_train_ext.shape[0], x_train.shape[0]))
    num_samples = x_train.shape[0]
    index_array = np.arange(num_samples)
    #np.random.shuffle(index_array)
  
    try:
        from keras.engine.training import _make_batches as make_batches
    except ImportError:
        from keras.engine.training_utils import make_batches
  
    sample_batch_size = 128
    batches = make_batches(num_samples, sample_batch_size)
  
    x_train_new = np.zeros((num_samples*2, x_train.shape[1]), dtype=np.float32)
    
    #MK 3 components
    y_train_new = [np.zeros((num_samples*2,), dtype=np.float32), np.zeros((num_samples*2,), dtype=np.float32),\
                   np.zeros((num_samples*2,), dtype=np.float32)]
  
    for batch_index, (batch_start, batch_end) in enumerate(batches):
        batch_ids = index_array[batch_start:batch_end]
        x_train_new[batch_start*2:batch_start*2 + (batch_end-batch_start)] = x_train[batch_ids]
        x_train_new[batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_x_train_ext[batch_ids]
        y_train_new[0][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[0][batch_ids]
        y_train_new[0][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[0][batch_ids]
        y_train_new[1][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[1][batch_ids]
        y_train_new[1][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[1][batch_ids]
        
        #MK
        y_train_new[2][batch_start*2:batch_start*2 + (batch_end-batch_start)] = y_train[2][batch_ids]
        y_train_new[2][batch_start*2 + (batch_end-batch_start):batch_end*2] = pu_y_train_ext[2][batch_ids]
    
    logger.info('Mixed muon data with pileup data. x_train_new has shape {0}, y_train_new has shape {1},{2},{3}'.format(x_train_new.shape, y_train_new[0].shape, y_train_new[1].shape, y_train_new[2].shape))
    return x_train_new, y_train_new

####################################################################
# MK
####################################################################

def MKdata_split(x,y,w,x_mask):
    # Split dataset in training and testing
    x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test = train_test_split(x, y, w, x_mask, test_size=test_size)
    logger.info('Loaded # of training and testing events: {0}'.format((x_train.shape[0], x_test.shape[0])))
  
    # Check for cases where the number of events in the last batch could be too few
    validation_split = 0.1
    train_num_samples = int(x_train.shape[0] * (1.0-validation_split))
    val_num_samples = x_train.shape[0] - train_num_samples
    batch_size = 128
    if (train_num_samples%batch_size) < 100:
        logger.warning('The last batch for training could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
        logger.warning('Try this formula: int(int({0}*{1})*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
    train_num_samples = int(x_train.shape[0] * 2 * (1.0-validation_split))
    val_num_samples = x_train.shape[0] - train_num_samples
    batch_size = 128
    if (train_num_samples%batch_size) < 100:
        logger.warning('The last batch for training after mixing could be too few! ({0}%{1})={2}. Please change test_size.'.format(train_num_samples, batch_size, train_num_samples%batch_size))
        logger.warning('Try this formula: int(int({0}*{1})*2*{2}) % 128'.format(x.shape[0], 1.0-test_size, 1.0-validation_split))
    return x_train, x_test, y_train, y_test, w_train, w_test, x_mask_train, x_mask_test
