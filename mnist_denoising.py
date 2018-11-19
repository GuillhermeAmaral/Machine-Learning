"""
Created on Sun Nov 18 19:56:10 2018

@author: Guillherme

Stacked Autoencoder for denoising images (mnist dataset)

Source: https://blog.keras.io/building-autoencoders-in-keras.html?fbclid=IwAR1g_unok9zikI1bdf4Cox5HX6lxMpW9zbLwX91jhNsYodHLbgL_dZ_BfCM
"""

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Dense
import numpy as np
import matplotlib.pyplot as plt

# Obtenção do dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Normalização
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Adição do ruído 
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Plot, Linha 1 imagem original e linha 2 a imagem com ruído
# Apenas as 10 primeiras imagens
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noisy
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

# Modelo
autoencoder = Model(input_img, decoded)
# O otimizador ajusta a taxa de aprendizagem
# A função para calcular o loss é a binary_crossentropy
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# Executa o modelo
autoencoder.fit(x_train_noisy, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

W = autoencoder.get_weights()
np.save('SAE_W.npy', W)
# A linha abaixo inicializa a rede com os pesos de um treinamento anterior
#W = np.load('SAE_W.npy')
#autoencoder.set_weights(W)

# this model maps an input to its encoded representation
# Encoder, representação em 32 pixels da imagem
encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)

encoded_input = Input(shape=(32,))
decoder = Model(encoded_input, autoencoder.layers[-3](encoded_input))
decoded_imgs = decoder.predict(encoded_imgs)

encoded_input = Input(shape=(64,))
decoder = Model(encoded_input, autoencoder.layers[-2](encoded_input))
decoded_imgs = decoder.predict(decoded_imgs)

encoded_input = Input(shape=(128,))
decoder = Model(encoded_input, autoencoder.layers[-1](encoded_input))
decoded_imgs = decoder.predict(decoded_imgs)

# PLOT em 3 linhas
# Linha 1 é a imagem original, linha 2 a imagem com ruído e linha 3 a imagem processada
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noise
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()