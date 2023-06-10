from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

import argparse
import json
from keras import backend as K
from keras.losses import categorical_crossentropy
from keras.layers import Layer
from keras.layers import Input, LSTM, TimeDistributed
from keras.layers.core import Dense, Lambda
from keras.models import Model
from nltk.tokenize import word_tokenize
from ocpa.algo.conformance.precision_and_fitness import evaluator as quality_measure_factory
import tensorflow as tf
import pandas as pd
import numpy as np
import codecs
import os, re
from tqdm import tqdm
import random
from datetime import datetime, timedelta
import time


def create_lstm_vae(input_dim,
                    batch_size,  # we need it for sampling
                    intermediate_dim,
                    latent_dim):
    """
    Creates an LSTM Variational Autoencoder (VAE).

    # Arguments
        input_dim: int.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(None, input_dim,))

    # LSTM encoding
    h = LSTM(units=intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(units=latent_dim)(h)
    z_log_sigma = Dense(units=latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.0)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    z_reweighting = Dense(units=intermediate_dim, activation="linear")
    z_reweighted = z_reweighting(z)

    # "next-word" data for prediction
    decoder_words_input = Input(shape=(None, input_dim,))

    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True, return_state=True)

    # todo: not sure if this initialization is correct
    h_decoded, _, _ = decoder_h(decoder_words_input, initial_state=[z_reweighted, z_reweighted])
    decoder_dense = TimeDistributed(Dense(input_dim, activation="softmax"))
    decoded_onehot = decoder_dense(h_decoded)

    # end-to-end autoencoder
    vae = Model([x, decoder_words_input], decoded_onehot)

    # encoder, from inputs to latent space
    encoder = Model(x, [z_mean, z_log_sigma])

    # generator, from latent space to reconstructed inputs -- for inference's first step
    decoder_state_input = Input(shape=(latent_dim,))
    _z_rewighted = z_reweighting(decoder_state_input)
    _h_decoded, _decoded_h, _decoded_c = decoder_h(decoder_words_input, initial_state=[_z_rewighted, _z_rewighted])
    _decoded_onehot = decoder_dense(_h_decoded)
    generator = Model([decoder_words_input, decoder_state_input], [_decoded_onehot, _decoded_h, _decoded_c])

    # RNN for inference
    input_h = Input(shape=(intermediate_dim,))
    input_c = Input(shape=(intermediate_dim,))
    __h_decoded, __decoded_h, __decoded_c = decoder_h(decoder_words_input, initial_state=[input_h, input_c])
    __decoded_onehot = decoder_dense(__h_decoded)
    stepper = Model([decoder_words_input, input_h, input_c], [__decoded_onehot, __decoded_h, __decoded_c])

    def vae_loss(x, x_decoded_onehot):
        xent_loss = categorical_crossentropy(x, x_decoded_onehot)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer="adam", loss=vae_loss)
    vae.summary()

    return vae, encoder, generator, stepper


def decode_sequence(states_value, decoder_adapter_model, rnn_decoder_model, num_decoder_tokens, token2id, id2token, max_seq_length):
    """
    Decoding adapted from this example:
    https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

    :param states_value:
    :param decoder_adapter_model: reads text representation, makes the first prediction, yields states after the first RNN's step
    :param rnn_decoder_model: reads previous states and makes one RNN step
    :param num_decoder_tokens:
    :param token2id: dict mapping words to ids
    :param id2token: dict mapping ids to words
    :param max_seq_length: the maximum length of the sequence
    :return:
    """

    # generate empty target sequence of length 1
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # populate the first token of the target sequence with the start character
    target_seq[0, 0, token2id["\t"]] = 1.0

    # sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1)
    stop_condition = False
    decoded_sentence = ""

    first_time = True
    h, c = None, None

    while not stop_condition:

        if first_time:
            # feeding in states sampled with the mean and std provided by encoder
            # and getting current LSTM states to feed in to the decoder at the next step
            output_tokens, h, c = decoder_adapter_model.predict([target_seq, states_value])
            first_time = False
        else:
            # reading output token
            output_tokens, h, c = rnn_decoder_model.predict([target_seq, h, c])

        # sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = id2token[sampled_token_index]
        decoded_sentence += sampled_token + " "

        # exit condition: either hit max length
        # or find stop character.
        if sampled_token == "<end>" or len(decoded_sentence) > max_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

    return decoded_sentence

def get_text_data(data_path, num_samples=1000):
    """
    Reads the data from the file and vectorizes it
    :param data_path: path to the data file
    :param num_samples: number of samples to read
    :return: input_texts, input_characters, num_encoder_tokens, max_encoder_seq_length
    """

    # vectorize the data
    input_texts = []
    input_characters = set(["\t"])

    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.read().lower().split("\n")

    for line in lines[: min(num_samples, len(lines) - 1)]:

        #input_text, _ = line.split("\t")
        input_text = word_tokenize(line)
        input_text.append("<end>")

        input_texts.append(input_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)

    input_characters = sorted(list(input_characters))
    num_encoder_tokens = len(input_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts]) + 1

    print("Number of samples:", len(input_texts))
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Max sequence length for inputs:", max_encoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())

    encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    decoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32")

    for i, input_text in enumerate(input_texts):
        decoder_input_data[i, 0, input_token_index["\t"]] = 1.0

        for t, char in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[char]] = 1.0
            decoder_input_data[i, t + 1, input_token_index[char]] = 1.0

    return max_encoder_seq_length, num_encoder_tokens, input_characters, input_token_index, reverse_input_char_index, \
           encoder_input_data, decoder_input_data


def VAE_generalization(ocel_gen, ocpn):
    """
    Calculates the generalization of the generated object-centric event log from the VAE
    :param ocel_gen: generated object-centric event log
    :param ocpn: original object-centric process net
    :return: generalization
    """
    # use precision and fitness from the ocpa package
    precision, fitness = quality_measure_factory.apply(ocel_gen, ocpn)
    print("Precision of IM-discovered net: ",np.round(precision,4))
    print("Fitness of IM-discovered net: ", np.round(fitness,4))
    # calculate generalization as harmonic mean of precision and fitness
    generalization = 2 * ((fitness * precision) / (fitness + precision))
    print("VAE Generalization=", np.round(generalization,4))

    return generalization


def create_VAE_input(ocel, save_path=None):
    # we create another dictionary that only contains the the value inside the list to be able to derive the case
    mapping_dict = {key: ocel.process_execution_mappings[key][0] for key in ocel.process_execution_mappings}

    # we generate a new column in the class (log) that contains the process execution (case) number via the generated dictionary
    ocel.log.log['event_execution'] = ocel.log.log.index.map(mapping_dict)
    # Sort the DataFrame by 'event_timestamp' within each group
    sorted_df = ocel.log.log.groupby('event_execution').apply(lambda x: x.sort_values('event_timestamp'))

    # Reset the index to remove the index level
    sorted_df = sorted_df.reset_index(drop=True)

    # Concatenate the 'event_activity' values within each group into a single string
    train_log = sorted_df.groupby('event_execution')['event_activity'].apply(lambda x: ' '.join(x)).tolist()
    # save the file if a save_path is given to retrieve it again
    if save_path is not None:
        with open(save_path, "w") as file:
            for sentence in train_log:
                line = "".join(sentence) + "\n"
                file.write(line)
    return train_log