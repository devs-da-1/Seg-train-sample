import tensorflow as tf
from .configuration import get_config
from .metrics import dice_coef
from .loss import BceDiceLoss
from .utils import Timer
from .callbacks import get_callbacks

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

def init_model(model):
  config = get_config()
  model.compile(
      optimizer=config.get('opt')(learning_rate=config.get('lr')),
      # loss='binary_crossentropy',
      loss=BceDiceLoss(weights=[0.7,0.3]),
      weighted_metrics=[
          dice_coef,
          tf.keras.metrics.BinaryIoU(target_class_ids=[1])          
      ]
  )
  
  model.summary()

  return model

def run_experiment(model, train, val, test, epochs=5):
  config = get_config()
  callbacks, name = get_callbacks(val)
  timer = Timer()
  timer.begin()
  model.fit(
      train.data,
      epochs=epochs,
      steps_per_epoch=train.size//config.get('batch_size'),
      validation_data=val.data,
      validation_steps=val.size//config.get('batch_size'),
      callbacks=callbacks
  )
  timer.end()

  return name
