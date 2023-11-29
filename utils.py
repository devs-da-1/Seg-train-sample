from .configuration import get_config
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time

def viz_images(ds):
  config = get_config()
  image_batch, mask_batch = next(iter(ds.data))
  rows = np.min([config.get('batch_size'), 6])
  fig, ax = plt.subplots(ncols=rows,nrows=2,figsize=(15, 5))
  for i in range(rows):
    ax[0][i].imshow(image_batch[i].numpy().astype("float32"),vmin=0,vmax=1,cmap='gray')
    ax[0][i].axis("off")

    ax[1][i].imshow(mask_batch[i].numpy().astype("float32"),vmin=0,vmax=1,cmap='gray')
    ax[1][i].axis("off")
  plt.suptitle('Imágenes vs Máscaras', fontsize=20)
  plt.show()
  plt.close('all')

class Timer():
  def begin(self):
    self.start = time.time()
  
  def end(self):
    total_time = time.time() - self.start
    minutes = int(total_time // 60)
    seconds = np.round(total_time % 60)
    
    print(f'Time: {minutes} minutes {seconds} seconds.\n')

def viz_predictions(model, ds, num_samples=3):
  config = get_config()
  # Predict
  counter = 1
  for sample in ds.data:
    img, msk = sample
    print(img.shape, msk.shape)
    y_pred = model.predict(img,batch_size=config.get('batch_size'))
    print(y_pred.shape)
    
    for counter in range(num_samples):#config.get('batch_size')):
      fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 5))
      ax[0].imshow(tf.image.resize(img[counter], config.get("mask_shape")))
      ax[0].set_title('Imagen')
      ax[0].axis("off")

      ax[1].imshow(msk[counter], cmap='Blues',vmin=0.0,vmax=1.0)
      ax[1].set_title('Máscara')
      ax[1].axis("off")

      ax[2].imshow(y_pred[counter], cmap='Reds',vmin=0.0,vmax=1.0)
      ax[2].set_title('Predicción')
      ax[2].axis("off")

      ax[3].imshow(tf.math.round(y_pred[counter]), cmap='Reds',vmin=0.0,vmax=1.0)
      ax[3].set_title('Predicción binarizada')
      ax[3].axis("off")

      plt.show()
      plt.close('all')

    break