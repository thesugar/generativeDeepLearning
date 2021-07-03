# %%
from IPython import get_ipython

# %%
# VAE training
## imports
get_ipython().run_line_magic('load_ext', "autoreload")
get_ipython().run_line_magic('autoreload', '2')
import os
from models.VAE import VariationalAutoencoder
from utils.loaders import load_mnist

# %%
# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = f'run/{SECTION}/'
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

mode = 'build' # 'load'

# %%
(x_train, y_train), (x_test, y_test) = load_mnist()

# %%
# architecture
vae = VariationalAutoencoder(
    input_dim=(28, 28, 1)
    , encoder_conv_filters=[32, 64, 64, 64]
    , encoder_conv_kernel_size=[3, 3, 3, 3]
    , encoder_conv_strides=[1, 2, 2, 1]
    , decoder_conv_t_filters= [64, 64, 32, 1]
    , decoder_conv_t_kernel_size=[3, 3, 3, 3]
    , decoder_conv_t_strides=[1, 2, 2, 1]
    , z_dim=2
    , r_loss_factor=1000
)

if mode == 'build':
    vae.save(RUN_FOLDER)
else:
    vae.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# %%
vae.encoder.summary()

# %%
vae.decoder.summary()

# %%
# training
LEARNING_RATE = 0.0005

# %%
vae.compile(LEARNING_RATE)

# %%
BATCH_SIZE = 32
EPOCHS = 200
PRINT_EVERY_N_BATCHES = 100
INITIAL_EPOCH = 0

# %%
# このセルは実行できたものの処理が重くて実行結果セルの保存ができなかったため output の出力は省略
# 各エポックごとに出力された画像ファイルは保存済み
vae.train(
    x_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches= PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
)
# %%
