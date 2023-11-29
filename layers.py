import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, MaxPooling2D,\
                                    Conv2DTranspose, Cropping2D

def Conv(x, filters):
  x = Conv2D(
      filters=filters,
      kernel_size=(3,3),
      activation='relu',
      padding='same',
      kernel_initializer='he_normal'
  )(x)
  x = Dropout(0.1)(x)
  return x

def FinalConv(x, filters):
  x = Conv2D(
      filters=filters,
      kernel_size=(1,1),
      activation='sigmoid',
      kernel_initializer='he_normal'
  )(x)
  return x

def ConvTrans(x, filters):
  x = Conv2DTranspose(
      filters=filters,
      kernel_size=(3,3),
      strides=(2,2),
      activation='relu',
      padding='same',
      kernel_initializer='he_normal'
  )(x)
  return x

def Pooling(x):
  x = MaxPooling2D(
      pool_size=(2,2)
  )(x)
  return x

def Concat(x, skip):
  # Crop skip input connection
  x_shape = x.shape
  sk_shape = skip.shape
  cropping = []
  for i in range(1,3):
    c = sk_shape[i] - x_shape[i]
    if c%2==0:
      cropping.append((c//2,c//2))
    else:
      cropping.append((c//2,c//2+1))
  
  skip = Cropping2D(
      cropping=tuple(cropping)
  )(skip)

  # Concatenate filters
  x = tf.concat([x,skip], axis=-1)
  return x
