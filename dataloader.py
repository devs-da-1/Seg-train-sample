import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from .configuration import get_config
from pathlib import Path
import imghdr
import random
from glob import glob

def get_filepaths():
  config = get_config()
  images = glob('./data/images/*.png')
  masks = glob('./data/masks/*.png')

  num_samples = len(images)

  lst = list(zip(images, masks))
  random.Random(config.get('seed')).shuffle(lst)

  images, masks = list(zip(*lst))

  train_prc = config.get('train_prc')
  val_prc = config.get('val_prc')

  train_img = list(images[:int(num_samples*train_prc)])
  train_msk = list(masks[:int(num_samples*train_prc)])

  val_img = list(images[int(num_samples*train_prc):int(num_samples*val_prc)])
  val_msk = list(masks[int(num_samples*train_prc):int(num_samples*val_prc)])

  test_img = list(images[int(num_samples*val_prc):])
  test_msk = list(masks[int(num_samples*val_prc):])

  print('\ntrain:',len(train_img))
  print('val:',len(val_img))
  print('test:',len(test_img),'\n')

  return [
    (train_img, train_msk),
    (val_img, val_msk),
    (test_img, test_msk)
    ]

def decode_img(img, channels, img_type):
  config = get_config()
  if img_type != 'mask':
    img_height, img_width = config.get('input_shape')
    img = tf.io.decode_png(img, channels=channels)
  else:
    img_height, img_width = config.get('mask_shape')
    img = tf.io.decode_png(img, channels=channels)
  
  # Resize the image to the desired size
  return tf.image.resize(img, [img_height, img_width])

def process_path(file_path, channels, img_type):
  # Load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img, channels, img_type)
  if img_type != 'mask':
    img = tf.cast(img,tf.float32)/255.0
  else:
    img = tf.cast(img > 0,tf.float32)

  return img

def configure_for_performance(ds):
  config = get_config()
  ds = ds.cache()
  ds = ds.shuffle(buffer_size=config.get('batch_size'))
  ds = ds.repeat()
  ds = ds.batch(config.get('batch_size'))
  return ds

@tf.function
def data_aug(input_image, input_mask):
  config = get_config()

  input_shape = config.get('input_shape')
  mask_shape = config.get('mask_shape')

  # zoom in a bit
  if tf.random.uniform(()) > 0.5:
      # use original image to preserve high resolution
      input_image = tf.image.central_crop(input_image, 0.75)
      input_mask = tf.image.central_crop(input_mask, 0.75)
      # resize
      input_image = tf.image.resize(input_image, input_shape)
      input_mask = tf.image.resize(input_mask, mask_shape)
  
  # flipping random horizontal or vertical
  if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_left_right(input_image)
      input_mask = tf.image.flip_left_right(input_mask)
  if tf.random.uniform(()) > 0.5:
      input_image = tf.image.flip_up_down(input_image)
      input_mask = tf.image.flip_up_down(input_mask)

  # rotation in 30° steps
  rot_factor = tf.cast(tf.random.uniform(shape=[], maxval=12, dtype=tf.int32), tf.float32)
  angle = np.pi/12*rot_factor
  input_image = tfa.image.rotate(input_image, angle,fill_mode='reflect')
  input_mask = tfa.image.rotate(input_mask, angle,fill_mode='reflect')

  return input_image, input_mask

def get_ds(img, msk, train=False):
  img_ds = tf.data.Dataset.from_tensor_slices(img).map(lambda x: process_path(x,3,'img'))
  msk_ds = tf.data.Dataset.from_tensor_slices(msk).map(lambda x: process_path(x,1,'mask'))
  ds = tf.data.Dataset.zip((img_ds, msk_ds))
  ds = configure_for_performance(ds)
  if train:
    # Data augmentation
    ds = ds.map(data_aug)
  return ds

class Dataset():
  def __init__(self, img_list, msk_list, train=False):
    self.size = len(img_list)
    self.data = get_ds(img_list, msk_list, train=train)


def load_data():
  train, val, test = get_filepaths()
  train = Dataset(train[0], train[1], train=True)
  val = Dataset(val[0], val[1])
  test = Dataset(test[0], test[1])

  return train, val, test
