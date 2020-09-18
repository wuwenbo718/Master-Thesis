#MIT License
#
#Copyright (c) 2017 Erik Linder-Norén
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
# Laboratory of Robotics and Cognitive Science
# Version by:  Rafael Anicet Zanini
# Github:      https://github.com/larocs/EMG-GAN

from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, GRU
from tensorflow.keras.layers import Add, Subtract, Multiply, ReLU, ThresholdedReLU, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAvgPool1D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputSpec, Layer
from tensorflow.keras import initializers, regularizers, constraints
import numpy as np
import pywt
from tensorflow.compat.v1 import spectral

import warnings 
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

# From a PR that is not pulled into Keras
# https://github.com/fchollet/keras/pull/3677

class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.

    # Example

    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)

        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)

        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```

    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.

    # Input shape
        2D tensor with shape: `(samples, input_dim)`.

    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.

    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Discriminator():
    def __init__(self, args, training = False):
        self.channels = args['channels']
        self.conv_activation = args['conv_activation']
        self.num_steps = args['num_steps']
        self.seq_shape = (self.num_steps, self.channels)
        self.training_mode = training

        self.dropout_rate = args['dropout_rate']  # Dropout rate

        # Define parameters for wavelet decomposition
        self.wavelet_mother = args['wavelet_mother'] 
        self.wavelet_levels = args['wavelet_levels'] 
        self.wavelet_trainable= args['wavelet_trainable']
        self.use_mini_batch = args['use_mini_batch']

        self.sliding_window = args['sliding_window']
        self.activation_function = args["activation_function"]
        self.moving_avg_window = args["moving_avg_window"]
        self.model = self.build_critic()
        
    def make_wavelet_expansion(self, input_tensor):
        #input_tensor = Reshape((input_tensor.shape[-1],1))(input_tensor)
        print(input_tensor.shape)
        low_pass, high_pass  = pywt.Wavelet(self.wavelet_mother).filter_bank[:2]
        low_pass_filter = np.array(low_pass)
        high_pass_filter = np.array(high_pass)
        n_levels = self.wavelet_levels
        trainable=self.wavelet_trainable
        
        wv_kwargs = {
            "filters":1,
            "kernel_size":len(low_pass),
            "strides":2, 
            "use_bias":False, 
            "padding":"same", 
            "trainable":trainable,
        }

        approximation_coefficients = []
        detail_coefficients = []

        last_approximant = input_tensor
        last_approximant = K.reshape(last_approximant,(-1,last_approximant.shape[-2],last_approximant.shape[-1],1))
        for i in range(n_levels):
            lpf = low_pass_filter
            hpf = high_pass_filter
            #print(lpf.reshape((-1, int(self.channels))).shape)
            for j in range(self.channels):
                if j == 0:
                    a_n = Conv1D(
                        kernel_initializer=keras.initializers.Constant(lpf.reshape((-1, 1))),
                        name="low_pass_{}.{}".format(i,j),
                        **wv_kwargs
                    )(last_approximant[:,:,j,:])
                    d_n = Conv1D(
                        kernel_initializer=keras.initializers.Constant(hpf.reshape((-1, 1))),
                        name="high_pass_{}.{}".format(i,j),
                        **wv_kwargs,
                    )(last_approximant[:,:,j,:])
                else:
                #print(a_n)
                    temp_a = Conv1D(
                        kernel_initializer=keras.initializers.Constant(lpf.reshape((-1, 1))),
                        name="low_pass_{}.{}".format(i,j),
                        **wv_kwargs
                    )(last_approximant[:,:,j,:])
                    a_n = K.concatenate([a_n,temp_a],axis=2)
                    temp_d = Conv1D(
                        kernel_initializer=keras.initializers.Constant(hpf.reshape((-1, 1))),
                        name="high_pass_{}.{}".format(i,j),
                        **wv_kwargs,
                    )(last_approximant[:,:,j,:])
                    d_n = K.concatenate([d_n,temp_d],axis=2)
            detail_coefficients.append(d_n)
            approximation_coefficients.append(a_n)
            last_approximant = a_n
            last_approximant = K.reshape(last_approximant,(-1,last_approximant.shape[-2],last_approximant.shape[-1],1))

        return approximation_coefficients, detail_coefficients
    
    def envelopes(self, args):
        """Reparameterization trick by sampling fr an isotropic unit Gaussian.

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        input_ = args
        abs_envelope = K.abs(input_)
        envelope = tf.signal.frame(
            abs_envelope,
            self.moving_avg_window,
            1,#steps
            pad_end=True,
            pad_value=0,
            axis=1,
            name='envelope_moving_average'
        )
        #print(envelope.shape)
        #envelope_reshaped = K.reshape(envelope,(-1,self.num_steps,self.moving_avg_window))
        envelope_mean = K.mean(envelope, axis=2, keepdims=True)
        #print(envelope_mean.shape)
        envelope_mean = K.reshape(envelope_mean,(-1,self.num_steps,self.channels))
        return envelope_mean
    
    def rfft_layer(self,data):
        #data=Reshape((data.shape[1],data.shape[2],1))(data)
        n = data.shape[2]
        #print(data.shape)
        for i in range(n):
            if i == 0:
                fft = K.abs(tf.signal.rfft(data[:,:,i]))
                fft = K.reshape(fft,(-1,fft.shape[1],1))
            else:
                temp = K.abs(tf.signal.rfft(data[:,:,i]))
                temp = K.reshape(temp,(-1,temp.shape[1],1))
                fft = K.concatenate([fft,temp],axis=2)
        #print(fft.shape)
        return fft
    
    def build_critic(self):

        input_ = Input(shape=self.seq_shape)
        input_r = Reshape((self.seq_shape[0]*self.seq_shape[1],1))(input_)
        
        flat = Flatten()(input_)
        
        #CNN on raw signal
        cnn_1 = Conv1D(16, kernel_size=3, strides=2, padding="same", name='raw_conv_1')(input_)        
        cnn_1 = LeakyReLU(alpha=0.2)(cnn_1)
        cnn_1 = Dropout(self.dropout_rate)(cnn_1)
        cnn_2 = Conv1D(32, kernel_size=3, strides=2, padding="same", name='raw_conv_2')(cnn_1)
        cnn_2 = BatchNormalization(momentum=0.8)(cnn_2)
        cnn_2 = LeakyReLU(alpha=0.2)(cnn_2)
        cnn_2 = Dropout(self.dropout_rate)(cnn_2)
        #cnn_2 = MaxPooling2D(2)(cnn_2)
        cnn_3 = Conv1D(64, kernel_size=3, strides=2, padding="same", name='raw_conv_3')(cnn_2)
        cnn_3 = BatchNormalization(momentum=0.8)(cnn_3)
        cnn_3 = LeakyReLU(alpha=0.2)(cnn_3)
        cnn_3 = Dropout(self.dropout_rate)(cnn_3)
        #cnn_3 = MaxPooling2D(2)(cnn_3)
        cnn_4_out = Conv1D(32, kernel_size=3, strides=2, padding="same", name='raw_conv_4')(cnn_3)
        cnn_4 = BatchNormalization(momentum=0.8)(cnn_4_out)
        cnn_4 = LeakyReLU(alpha=0.2)(cnn_4)
        cnn_4 = Dropout(self.dropout_rate)(cnn_4)
        cnn_4 = Flatten()(cnn_4)
        
        #CNN on FFT of raw signal
        fft = Lambda(self.rfft_layer,name='rfft')(input_)
        #fft = Reshape((fft.shape[1]*fft.shape[2],1))(fft)
        #fft_abs = Lambda(K.abs)(fft)
        #fft_abs = Reshape((fft_abs.shape[-1],1), name='fft_abs')(fft_abs)
        #fft_abs = Reshape((fft.shape[-2],fft.shape[-1],1), name='fft_abs')(fft)
        fft_cnn_1 = Conv1D(16, kernel_size=3, strides=2, padding="same", name='fft_conv_1')(fft)
        fft_cnn_1 = LeakyReLU(alpha=0.2)(fft_cnn_1)
        fft_cnn_1 = Dropout(self.dropout_rate)(fft_cnn_1)
        #fft_cnn_1 = MaxPooling2D(2)(fft_cnn_1)
        fft_cnn_2 = Conv1D(32, kernel_size=3, strides=2, padding="same", name='fft_conv_2')(fft_cnn_1)
        fft_cnn_2 = BatchNormalization(momentum=0.8)(fft_cnn_2)
        fft_cnn_2 = LeakyReLU(alpha=0.2)(fft_cnn_2)
        fft_cnn_2 = Dropout(self.dropout_rate)(fft_cnn_2)
        #fft_cnn_2 = MaxPooling2D(2)(fft_cnn_2)
        fft_cnn_3 = Conv1D(64, kernel_size=3, strides=2, padding="same", name='fft_conv_3')(fft_cnn_2)
        fft_cnn_3 = BatchNormalization(momentum=0.8)(fft_cnn_3)
        fft_cnn_3 = LeakyReLU(alpha=0.2)(fft_cnn_3)
        fft_cnn_3 = Dropout(self.dropout_rate)(fft_cnn_3)
        #fft_cnn_3 = MaxPooling2D(2)(fft_cnn_3)
        fft_cnn_4_out = Conv1D(64, kernel_size=3, strides=2, padding="same", name='fft_conv_4')(fft_cnn_3)
        fft_cnn_4 = BatchNormalization(momentum=0.8)(fft_cnn_4_out)
        fft_cnn_4 = LeakyReLU(alpha=0.2)(fft_cnn_4)
        fft_cnn_4 = Dropout(self.dropout_rate)(fft_cnn_4)        
        fft_cnn_4 = Flatten()(fft_cnn_4)
                
        #CNN on FFT of envelope
        envelope_window = Lambda(self.envelopes, output_shape=(self.seq_shape[0],self.seq_shape[1]), name='envelope')(input_)
        #print(envelope_window.shape)
        #envelope_window = Flatten()(envelope_window)
        envelope_fft = Lambda(self.rfft_layer,name='envelope_fft')(envelope_window)
        #envelope_fft_abs = Lambda(K.abs)(envelope_fft)
        #envelope_fft_abs = Reshape((envelope_fft_abs.shape[-1],1))(envelope_fft_abs)
        #envelope_fft_abs = Reshape((envelope_fft.shape[-2],envelope_fft.shape[-1],1))(envelope_fft)
        #envelope_fft =Reshape((envelope_fft.shape[1]*envelope_fft.shape[2],1))(envelope_fft)
        envelope_cnn_1 = Conv1D(16, kernel_size=3, strides=2, padding="same", name='fft_env_conv_1')(envelope_fft)
        envelope_cnn_1 = LeakyReLU(alpha=0.2)(envelope_cnn_1)
        envelope_cnn_1 = Dropout(self.dropout_rate)(envelope_cnn_1)
        #envelope_cnn_1 = MaxPooling2D(2)(envelope_cnn_1)
        envelope_cnn_2 = Conv1D(32, kernel_size=3, strides=2, padding="same", name='fft_env_conv_2')(envelope_cnn_1)
        envelope_cnn_2 = BatchNormalization(momentum=0.8)(envelope_cnn_2)
        envelope_cnn_2 = LeakyReLU(alpha=0.2)(envelope_cnn_2)
        envelope_cnn_2 = Dropout(self.dropout_rate)(envelope_cnn_2)
        #envelope_cnn_2 = MaxPooling2D(2)(envelope_cnn_2)
        envelope_cnn_3 = Conv1D(64, kernel_size=3, strides=2, padding="same", name='fft_env_conv_3')(envelope_cnn_2)
        envelope_cnn_3 = BatchNormalization(momentum=0.8)(envelope_cnn_3)
        envelope_cnn_3 = LeakyReLU(alpha=0.2)(envelope_cnn_3)
        envelope_cnn_3 = Dropout(self.dropout_rate)(envelope_cnn_3)
        #envelope_cnn_3 = MaxPooling2D(2)(envelope_cnn_3)
        envelope_cnn_4_out = Conv1D(64, kernel_size=3, strides=2, padding="same", name='fft_env_conv_4')(envelope_cnn_3)
        envelope_cnn_4 = BatchNormalization(momentum=0.8)(envelope_cnn_4_out)
        envelope_cnn_4 = LeakyReLU(alpha=0.2)(envelope_cnn_4)
        envelope_cnn_4 = Dropout(self.dropout_rate)(envelope_cnn_4)
        envelope_cnn_4 = Flatten()(envelope_cnn_4)
        
        # Wavelet Expansion
        approx_stack, detail_stack = self.make_wavelet_expansion(input_)
        features_list = []
        features_list.extend(detail_stack)
        features_list.append(approx_stack[-1])
        wavelet_concatenate = Concatenate(axis=1,name='wavelet_concat')(features_list)
        wavelet_cnn_1 = Conv1D(16, kernel_size=3, strides=2, padding="same", name='wavelet_conv_1')(wavelet_concatenate)
        wavelet_cnn_1 = LeakyReLU(alpha=0.2)(wavelet_cnn_1)
        wavelet_cnn_1 = Dropout(self.dropout_rate)(wavelet_cnn_1)
        wavelet_cnn_2 = Conv1D(32, kernel_size=3, strides=2, padding="same", name='wavelet_conv_2')(wavelet_cnn_1)
        wavelet_cnn_2 = BatchNormalization(momentum=0.8)(wavelet_cnn_2)
        wavelet_cnn_2 = LeakyReLU(alpha=0.2)(wavelet_cnn_2)
        wavelet_cnn_2 = Dropout(self.dropout_rate)(wavelet_cnn_2)
        wavelet_cnn_3 = Conv1D(64, kernel_size=3, strides=2, padding="same", name='wavelet_conv_3')(wavelet_cnn_2)
        wavelet_cnn_3 = BatchNormalization(momentum=0.8)(wavelet_cnn_3)
        wavelet_cnn_3 = LeakyReLU(alpha=0.2)(wavelet_cnn_3)
        wavelet_cnn_3 = Dropout(self.dropout_rate)(wavelet_cnn_3)
        wavelet_cnn_4_out = Conv1D(32, kernel_size=3, strides=2, padding="same", name='wavelet_conv_4')(wavelet_cnn_3)
        wavelet_cnn_4 = BatchNormalization(momentum=0.8)(wavelet_cnn_4_out)
        wavelet_cnn_4 = LeakyReLU(alpha=0.2)(wavelet_cnn_4)
        wavelet_cnn_4 = Dropout(self.dropout_rate)(wavelet_cnn_4)
        wavelet_cnn_4 = Flatten()(wavelet_cnn_4)
        
        # Mini batch discrimination
        mini_disc = MinibatchDiscrimination(10,3)(flat)
        
        if self.use_mini_batch:
            concatenate = Concatenate()([cnn_4,fft_cnn_4,envelope_cnn_4,wavelet_cnn_4,mini_disc])
        else:
            concatenate = Concatenate()([cnn_4,fft_cnn_4,envelope_cnn_4,wavelet_cnn_4])
        #print(wavelet_cnn_4.shape)
        #dense1 = Dense(526,activation='relu',name='dense1')(concatenate)
        #dense1 = Dropout(self.dropout_rate)(dense1)
        #dense2 = Dense(256,activation='relu',name='dense2')(dense1)
        #dense2 = Dropout(self.dropout_rate)(dense2)
        #dense3 = Dense(126,activation='relu',name='dense3')(dense2)
        #dense3 = Dropout(self.dropout_rate)(dense3)
        mlp = Dense(3,activation='softmax')(concatenate)
                
        model = Model(input_, mlp)

        if self.training_mode:
            print('Critic model:')
            model.summary()
            
            file_name = './output/critic.png'
            plot_model(model, to_file=file_name, show_shapes = True)
        
        return model
    
    def save(self, index=-1):
        if index == -1:
            file_path = './saved_models/critic.h5'
        else:
            file_path = './saved_models/critic_' + str(index) + '.h5'
        self.model.save_weights(file_path)
    
    def load(self, index=-1):
        if index == -1:
            file_path = './saved_models/critic.h5'
        else:
            file_path = './saved_models/critic_' + str(index) + '.h5'
        self.model = self.build_critic()
        self.model.load_weights(file_path)
    
    def predict(self, args):
        return self.model.predict(args)