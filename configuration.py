from tensorflow.keras.optimizers import Nadam

def get_config():
  SEED = 2023
  BATCH_SIZE = 10
  config = dict(
    train_prc=0.7,
    val_prc=0.9,
    input_shape=(256,256),
    mask_shape=(256,256),
    batch_size = BATCH_SIZE,
    buffer_size = BATCH_SIZE,
    lr = 0.001,
    opt = Nadam,
    initializer = 'he_normal',
  )
  return config
