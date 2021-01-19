from sys import stdin
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.engine.training import Model
import matplotlib.pyplot as plt

from utils.callbacks import CustomCallback, step_decay_schedule

import numpy as np
import json
import os
import pickle

class Autoencoder():
    def __init__(self
        , input_dim
        , encoder_conv_filters
        , encoder_conv_kernel_size
        , encoder_conv_strides
        , decoder_conv_t_filters
        , decoder_conv_t_kernel_size
        , decoder_conv_t_strides
        , z_dim
        , use_batch_norm = False
        , use_dropout = False
        ):
        self.name = 'autoencoder'

        self.input_dim = input_dim
        self.encoder_conv_filters = encoder_conv_filters
        self.encoder_conv_kernel_size = encoder_conv_kernel_size
        self.encoder_conv_strides = encoder_conv_strides
        self.decoder_conv_t_filters = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides = decoder_conv_t_strides
        self.z_dim = z_dim

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()

    def _build(self):

        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoer_input')

        x = encoder_input

        for i in range(self.n_layers_encoder):
            conv_layer = Conv2D(
                filters = self.encoder_conv_filters[i]
                , kernel_size = self.encoder_conv_kernel_size[i]
                , strides = self.encoder_conv_strides[i]
                , padding = 'same'
                , name = 'encoder_conv_' + str(i)
            )

            x = conv_layer(x)

            x = LeakyReLU()(x)

            if self.use_batch_norm:
                x = BatchNormalization()(x)

            if self.use_dropout:
                x = Dropout(rate=0.25)(x)

        # Flattening する前の shape（バッチサイズにあたる [0]（インデックスが 0）は除く）を保持
        # これ（shape_before_flattening）は Decoder で使う
        # （ただ、必ずしも Encoder と Decoder は対称でなくてもよい（※）から、ここではこうしているというだけ）
        # ※ デコーダの最後の層からの出力はエンコーダへの入力と同じサイズである必要あり（損失関数で双方をピクセル単位で比べられるから）
        # なお、K.int_shape(x) は x の shape をタプルの形で返す。
        #  e.g.) (1, 2, 3)[1:] のようにすれば (2, 3) が返る。
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)

        # Flatten で平坦化したベクトルを 2 次元の潜在空間に接続する全結合層
        encoder_output = Dense(self.z_dim, name='encoder_output')(x)

        self.encoder = Model(encoder_input, encoder_output)

        ### THE DECODER

        # デコーダへの入力（潜在空間中の点）を定義する
        decoder_input = Input(shape=(self.z_dim, ), name='decoder_input')

        # 入力を Dense（全結合層）へ接続
        x = Dense(units=np.prod(shape_before_flattening))(decoder_input)

        # 最初の転置畳み込み層に入力できるテンソルの形に変換
        x = Reshape(target_shape=shape_before_flattening)(x)

        for i in range(self.n_layers_decoder):
            # 転置畳み込み層（Conv2DTranspose）では、strides=2 と設定すると、
            # 入力テンソルのサイズが高さも幅も 2 倍になる。
            # 転置畳み込み層では、strides パラメータは画像のピクセルの間を 0 でパディングする際の 0 の個数を決定する
            #（なお、普通の畳み込み層 Conv2D では、strides=2 とすれば入力テンソルのサイズは高さも幅も半分になる）
            # 各層でサイズを拡張して、元の画像のサイズ 28x28 まで戻そうとしている
            conv_t_layer = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i]
                , kernel_size = self.decoder_conv_t_kernel_size[i]
                , strides = self.decoder_conv_t_strides[i]
                , padding = 'same'
                , name = 'decoder_conv_t_' + str(i)
            )

            x = conv_t_layer(x)

            if i < self.n_layers_decoder - 1:
                x = LeakyReLU()(x)

                if self.use_batch_norm:
                    x = BatchNormalization()(x)

                if self.use_dropout:
                    x = Dropout()(x)

            else:
                x = Activation('sigmoid')(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)

        ### THE FULL AUTOENCODER
        # オートエンコーダへの入力は、エンコーダへの入力と同じ
        model_input = encoder_input
        # オートエンコーダからの出力は、エンコーダからの出力をデコーダに通したもの
        model_output = self.decoder(encoder_output)   ## ⚠️ ここ、model_output = decoder_output などとせず、こう書く！

        self.model = Model(model_input, model_output)

    def compile(self, learning_rate):
        self.learning_rate = learning_rate

        optimizer = Adam(lr=learning_rate)

        # これはカスタム損失関数として定義しているのだろう
        # これで compile 関数に渡せば以下の関数を損失関数として計算してくれる
        # ---
        # 損失関数は通常、元の画像と復元された画像の個々のピクセルの、
        # 平均二乗誤差の平方根（RMSE）または 2 値交差エントロピーのどちらを使う。
        # 2 値交差エントロピーは大きく間違った極端な予測に対して誤差を大きくする。
        # このためピクセルの予測が予測範囲の中央に寄る傾向があり、その結果、画像はおとなしいものになる。
        # その理由から、ここでは損失関数として RMSE を使うが、どちらが正しいとか間違っているということではない。
        def r_loss(y_true, y_pred):
            return K.mean(K.square(y_true - y_pred), axis=[1,2,3])  ## axis: axes to compute mean

        self.model.compile(optimizer=optimizer, loss=r_loss)

    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(name=folder)
            os.makedirs(os.path.join(folder, 'viz'))
            os.makedirs(os.path.join(folder, 'weights'))
            os.makedirs(os.path.join(folder, 'images'))

        # b はバイナリファイルとして扱うモード
        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim
                , self.encoder_conv_filters
                , self.encoder_conv_kernel_size
                , self.encoder_conv_strides
                , self.decoder_conv_t_filters
                , self.decoder_conv_t_kernel_size
                , self.decoder_conv_t_strides
                , self.z_dim
                , self.use_batch_norm
                , self.use_dropout
            ], f)

        self.plot_model(folder)

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def train(self, x_train, batch_size, epochs, run_folder, print_every_n_batches=100, initial_epoch=0, lr_decay=1):

        custom_callback = CustomCallback(run_folder, print_every_n_batches, initial_epoch, self)
        lr_sched = step_decay_schedule(initial_lr=self.learning_rate, decay_factor=lr_decay, step_size=1)

        checkpoint2 = ModelCheckpoint(os.path.join(run_folder, 'weights/weights.h5'), save_weights_only = True, verbose=1)
        callback_list = [checkpoint2, custom_callback, lr_sched]

        self.model.fit(
            x_train
            , x_train
            , batch_size
            , shuffle = True
            , epochs = epochs
            , initial_epoch = initial_epoch
            , callbacks = callback_list
        )

    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder, 'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.encoder, to_file=os.path.join(run_folder, 'viz/encoder.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.decoder, to_file=os.path.join(run_folder, 'viz/decoder.png'), show_shapes = True, show_layer_names = True)