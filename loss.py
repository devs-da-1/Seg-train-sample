import tensorflow as tf
import numpy as np
from .metrics import dice_coef

class BceDiceLoss(tf.keras.losses.Loss):
  def __init__(self, weights=[0.5,0.5]):
    super().__init__()
    if np.sum(weights) != 1.0:
      raise ValueError('weights must add to 1.')
    self.weights = weights
    self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

  def dice(self, y_true, y_pred, sample_weight):
    return 1 - dice_coef(y_true, y_pred)

  def __call__(self, y_true, y_pred, sample_weight=None):
    return self.weights[0] * self.bce(y_true, y_pred, sample_weight) +\
           self.weights[1] * self.dice(y_true, y_pred, sample_weight)
