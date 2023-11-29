import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=0.0001):
  intersection = tf.math.reduce_sum(tf.math.multiply(y_true, y_pred))
  dice = (2.0 * intersection + smooth)/(tf.math.reduce_sum(y_true) +\
          tf.math.reduce_sum(y_pred) + smooth)
  return dice
