# %%
import numpy as np

from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, LeakyReLU, Dropout, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

from tensorflow.keras.datasets import cifar10

# %%
NUM_CLASSES = 10

# %%
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# %%
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

y_train = to_categorical(y_train, NUM_CLASSES)
y_test  = to_categorical(y_test, NUM_CLASSES)

print(x_train[54, 12, 13, 1])
# %%
# architecture
input_layer = Input(shape=(32, 32, 3))

conv_layer_1 = Conv2D(
    filters = 10
    , kernel_size = (4, 4)
    , strides = 2
    , padding = 'same'
)(input_layer)

conv_layer_2 = Conv2D(
    filters = 20
    , kernel_size = (3, 3)
    , strides = 2
    , padding = 'same'
)(conv_layer_1)

flatten_layer = Flatten()(conv_layer_2)

output_layer = Dense(units=10, activation='softmax')(flatten_layer)

model = Model(input_layer, output_layer)
# %%
model.summary()
# %%
# Conv2D + Batch Normalization, Activation (Leaky ReLU), Dropout を使う
# BAD (Batch Normalization, Activation, Dropout) で覚える（Dropout は入れていない部分もある）
# Batch Normalization と Activation の順番は逆でもよい（決まったルールはない）
# 最近では過学習対策で Dropout を使わず Batch Normalization のみで対応することも増えている

input_layer = Input(shape=(32, 32, 3))

x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=32, kernel_size=3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Conv2D(filters=64, kernel_size=3, strides=2, padding='same')(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeakyReLU()(x)
x = Dropout(rate = 0.5)(x)

x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

revised_model = Model(input_layer, output_layer)
# %%
revised_model.summary()
# %%
# train
optimizer = Adam(lr = 0.0005)
revised_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# %%
revised_model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True, validation_data=(x_test, y_test))
# %%
len(revised_model.layers)
# %%
revised_model.layers[5].get_weights()
# %%
revised_model.layers[6].get_weights()
# %%
revised_model.evaluate(x_test, y_test)
