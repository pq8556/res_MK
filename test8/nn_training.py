import datetime
import sys


from nn_logging import getLogger
logger = getLogger()

from nn_models import save_my_model
# MK
import numpy as np
# ______________________________________________________________________________
# See https://stackoverflow.com/q/616645

class TrainingLog(object):
  def __init__(self):
    import os
    import sys
    import tempfile
    fd, name = tempfile.mkstemp(suffix='.txt', prefix='keras_output_', dir='.', text=True)
    self.file = os.fdopen(fd, 'w')
    self.name = name
    self.stdout = sys.stdout
  def __del__(self):
    self.file.close()
  def __enter__(self):
    sys.stdout = self
  def __exit__(self, type, value, traceback):
    sys.stdout = self.stdout
  def write(self, msg):
    self.file.write(msg)
  def flush(self):
    self.file.flush()


# ______________________________________________________________________________
def train_model(model, x, y, model_name='model', batch_size=None, epochs=1, verbose=1, callbacks=None,
                validation_split=0., shuffle=True, class_weight=None, sample_weight=None):
  start_time = datetime.datetime.now()
  logger.info('Begin training ...')

  with TrainingLog() as tlog:  # redirect sys.stdout
    history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                        validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)

  logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))

  save_my_model(model, name=model_name)
  return history


def separate_input(x):
    dnn_input = x[:,0:39]
    cnn_input0phi = x[:,39:44]
    cnn_input0theta = x[:,44:49]
    cnn_input0bend = x[:,49:54]
    
    cnn_input1phi = cnn_input0phi.reshape((len(x),5,1))
    cnn_input1theta = cnn_input0theta.reshape((len(x),5,1))
    cnn_input1bend = cnn_input0bend.reshape((len(x),5,1))

    cnn_input0 = np.insert(cnn_input1theta,[0],cnn_input1phi,axis=2)
    cnn_input = np.insert(cnn_input1bend,[0],cnn_input0,axis=2)
    print(dnn_input.shape, cnn_input.shape)
    print(cnn_input[:3])
    
    return [dnn_input, cnn_input] 
    


# ______________________________________________________________________________
def train_model_CNN(model, x, y, model_name='model', batch_size=None, epochs=1, verbose=1, callbacks=None,
                    validation_split=0., shuffle=True, class_weight=None, sample_weight=None):
    start_time = datetime.datetime.now()
    logger.info('separating DNN and CNN inputs')    
    logger.info('Begin training ...')
    # x = [dnn_input, cnn_input]
    with TrainingLog() as tlog:  # redirect sys.stdout
        history = model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks,
                          validation_split=validation_split, shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight)
  
    logger.info('Done training. Time elapsed: {0} sec'.format(str(datetime.datetime.now() - start_time)))
  
    save_my_model(model, name=model_name)
    return history
