# %%
# `brew install graphviz` が必要
import os
from utils.loaders import load_mnist
from models.AE import Autoencoder

# %%
SECTION = 'vae'
RUN_ID = '0001'
DATA_NAME = 'digits'
RUN_FOLDER = f'run/{SECTION}/'
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))

MODE = 'build' # 'load'

# %%
(x_train, y_train), (x_test, y_test) = load_mnist()

# %%
# Define the structure of the neural network
AE = Autoencoder(
    input_dim=(28, 28, 1)
    , encoder_conv_filters= [32, 64, 64, 64]  # これは shape ではなく、「4 層重ねること」（リストの長さ）と、各層のフィルターの枚数の指定を行っている
    , encoder_conv_kernel_size= [3, 3, 3, 3]  # こちらは各層の kernel_size を指定している
    , encoder_conv_strides= [1, 2, 2, 1]
    , decoder_conv_t_filters= [64, 64, 32, 1]
    , decoder_conv_t_kernel_size= [3, 3, 3, 3]
    , decoder_conv_t_strides= [1, 2, 2, 1]
    , z_dim=2
)

if MODE == 'build':
    AE.save(RUN_FOLDER)
else: # 'load'
    AE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))

# %%
AE.encoder.summary()

# %%
AE.decoder.summary()

# %%
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
INITIAL_EPOCH = 0

# %%
AE.compile(LEARNING_RATE)

# %%
AE.train(
    x_train[:1000]
    , batch_size= BATCH_SIZE
    , epochs = 200
    , run_folder= RUN_FOLDER
    , initial_epoch= INITIAL_EPOCH
)
