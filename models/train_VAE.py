from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from ocpa.algo.conformance.precision_and_fitness import evaluator as quality_measure_factory
import numpy as np
import pickle
import logging
import pandas as pd
import pandas as pd
import numpy as np


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


def process_log(gen_log, ocel, ocpn, save_path = None):
    """
    This function takes a generated log and converts it to a log that can be used by the ocel object
    :param gen_log: the generated log
    :param ocel: the original ocel object where the sampled log is based on
    :param ocpn: the original ocpn object where the sampled log is based on
    :return: the final log as pandas dataframe
    """
    # generate mapping dictionary for activities and objects
    mapping_dict = {}
    for transition in ocpn.transitions:
        if not transition.silent:
            mapping_dict[transition.name] = list(transition.preset_object_type)
    # get the unique original activities in the log
    activities = np.unique(ocel.log.log.event_activity)

    # Create a dictionary mapping modified strings to original values
    mapping_dict_values = {activity.replace(' ', '').lower(): activity.lower() for activity in activities}

    new_log = []
    for inner_list in gen_log:
        inner_list = inner_list[0].split()  # Split the single string into individual words
        adjusted_inner_list = [mapping_dict_values.get(word, word) for word in inner_list]
        adjusted_string = ' '.join(adjusted_inner_list)
        new_log.append([adjusted_string])  # Wrap the reversed string in a new list

    original_log = []
    # Iterate over each inner list in gen_log
    for inner_list in new_log:
        # Create an empty list to store the original activity names
        original_inner_list = []

        # Split the concatenated activities into individual words
        activities_in_inner_list = inner_list[0].split()

        # Initialize the index variable
        i = 0
        # Loop through the activities_in_inner_list
        while i < len(activities_in_inner_list):
            # Initialize a variable to store the matched activity
            matched_activity = None
            # Loop through the activities list to find a match
            for activity in activities:
                # Check if the concatenated words match the activity
                if ' '.join(activities_in_inner_list[i:i + len(activity.split())]).lower() == activity.lower():
                    # Set the matched activity
                    matched_activity = activity
                    # Break the loop as a match is found
                    break

            # Check if a match is found
            if matched_activity:
                # Append the matched activity to the original_inner_list
                original_inner_list.append(matched_activity)
                # Update the index by the number of words in the matched activity
                i += len(matched_activity.split())
            else:
                # If no match is found, update the index by 1 to move to the next word
                i += 1

        # Append the original_inner_list to the original_log
        original_log.append(original_inner_list)
    # Create an empty list to store the data
    data = []

    # Define the start and end date in 2022
    start_date = datetime(2022, 1, 1)

    # Variables for tracking the date and count
    current_date = start_date
    count = 0

    # Iterate over each inner list in original_log
    for inner_list in original_log:

        # Get the activities in the inner list
        activities = inner_list

        # Generate a random timestamp within the execution day
        execution_time = random.uniform(0, 1) * 24
        execution_timestamp = current_date + timedelta(hours=execution_time)

        # Create a dictionary to store the column values for each activity
        activity_dict = {
            'event_activity': [],
            'event_execution': [],
            'event_timestamp': []
        }

        # Add each activity, execution, and timestamp to the data list
        for activity in activities:

            activity_dict['event_activity'].append(activity)
            activity_dict['event_execution'].append(count + 1)
            activity_dict['event_timestamp'].append(execution_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f'))

            # Check if the count is a multiple of 100
            if count % 100 == 0:
                current_date += timedelta(days=1)  # Increase the date by 1

            # Increment the timestamp for the next activity within the same execution
            execution_timestamp += timedelta(minutes=1)
        count += 1
        # Get the distinct object types for the activities in the inner list
        object_types = set()
        for activity in activities:
            object_types.update(mapping_dict.get(activity, []))

        # Add columns for each distinct object type and initialize with empty values
        for object_type in object_types:
            activity_dict[object_type] = [''] * len(activities)

        # Update the column values with the corresponding objects
        for idx, activity in enumerate(activities):
            objects = mapping_dict.get(activity, [])
            for object_type in objects:
                activity_dict[object_type][idx] = [f"{object_type}{count}"]

        # Fill empty values in object columns with an empty list
        for object_type in object_types:
            activity_dict[object_type] = [value if value != '' else [] for value in activity_dict[object_type]]

        # Extend the data list with the activity_dict
        data.extend([{k: v[i] for k, v in activity_dict.items()} for i in range(len(activities))])

    # Convert the data list into a pandas DataFrame
    df_log = pd.DataFrame(data)
    # Include the index as a column
    df_log.reset_index(inplace=True)

    # Rename the index column to 'event_id'
    df_log.rename(columns={'index': 'event_id'}, inplace=True)

    if save_path is not None:
        df_log.to_csv(save_path, index=False)

    return df_log
def create_VAE_input(ocel, save_path=None):
    """
    Creates the input for the VAE when the training is done on the original log
    :param ocel: object-centric event log
    :param save_path: path to save the input
    :return: input train_log for the VAE
    """
    # we create another dictionary that only contains the value inside the list to be able to derive the case
    mapping_dict = {key: ocel.process_execution_mappings[key][0] for key in ocel.process_execution_mappings}

    # we generate a new column in the class (log) that contains the process execution (case) number via the generated dictionary
    ocel.log.log['event_execution'] = ocel.log.log.index.map(mapping_dict)

    # Remove whitespaces from the event_activity column for better training
    ocel.log.log['event_activity_new'] = ocel.log.log['event_activity'].str.replace(' ', '')


    # Sort the DataFrame by 'event_timestamp' within each group
    sorted_df = ocel.log.log.groupby('event_execution').apply(lambda x: x.sort_values('event_timestamp'))

    # Reset the index to remove the index level
    sorted_df = sorted_df.reset_index(drop=True)

    # Concatenate the 'event_activity' values within each group into a single string
    train_log = sorted_df.groupby('event_execution')['event_activity_new'].apply(lambda x: ' '.join(x)).tolist()
    # save the file if a save_path is given to retrieve it again
    if save_path is not None:
        with open(save_path, "w") as file:
            for sentence in train_log:
                line = "".join(sentence) + "\n"
                file.write(line)
    return train_log
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

print("Loading data...")

ocel = pd.read_pickle('/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS4_log.pickle')


with open("/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/csv/DS4_ocpn.pickle", "rb") as file:
    ocpn = pickle.load(file)

print("Data loaded")


train_log = create_VAE_input(ocel,'../src/data/VAE_input/DS4.txt')

print("Data processed")


timesteps_max, enc_tokens, characters, char2id, id2char, x, x_decoder = get_text_data(num_samples=10000,
                                                                                      data_path='/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/VAE_input/DS4.txt')

print(x.shape, "Creating model...")

input_dim, timesteps = x.shape[-1], x.shape[-2]
batch_size, latent_dim = 1, 191
intermediate_dim, epochs = 353, 20

vae, enc, gen, stepper = create_lstm_vae(input_dim,
                                         batch_size=batch_size,
                                         intermediate_dim=intermediate_dim,
                                         latent_dim=latent_dim,
                                        )
print("Training model...")

vae.fit([x, x_decoder], x, epochs=epochs, verbose=1)

print("Fitted, predicting...")
#rearrange the input data and get the max amount of characters
max_length = max(len(string) for string in train_log)

def decode(s):
    return decode_sequence(s, gen, stepper, input_dim, char2id, id2char, max_length)

log = []

for _ in tqdm(range(500), desc="Sample Traces"):

    id_from = np.random.randint(0, x.shape[0] - 1)

    m_from, std_from = enc.predict([[x[id_from]]])

    seq_from = np.random.normal(size=(latent_dim,))
    seq_from = m_from + std_from * seq_from

    #print(decode(seq_from))
    log.append([decode(seq_from)])


df_log = process_log(log, ocel, ocpn, '/pfs/data5/home/ma/ma_ma/ma_nsabel/Generalization_in_Object_Centric_Process_Mining/src/data/VAE_generated/DS4_process_sampled.csv')