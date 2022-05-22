#########################################################
#
#   Drive_AI.py
#   2022 김연준 <kyjunclsrm@gmail.com>
#
#########################################################

import collections
import numpy as np
import pygame
import random
import os
import time
import math
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D, AveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from keras import initializers


from tensorflow.python.client import device_lib

# <State Vector>
# s_t[5]
#   [0]: distance to finish line / 200.0
#   [1]: distance to center / 200.0
#   [2]: agent rotation (rad, left rotation is positive)eeeeeeee
#   [3]: distance to obstacle / 200.0
#   [4]: direction to obstacle (rad)
#   (Discarded) [5]: obstacle size / 200.0

# <Q-Table Vector>
# a_t[3]
#   [0]: Turn Left
#   [1]: Move Straight
#   [2]: Turn Right


GAMMA = 0.99
INITIAL_EPSILON=0.5
FINAL_EPSILON = 0.0001
MEMORY_SIZE = 10000
NUM_EPOCHS_OBSERVE = 100
NUM_EPOCHS_TRAIN = 19900

LEARNING_RATE = 0.001
BATCH_SIZE = 32
BATCH_SIZE_1 = 32#24
BATCH_SIZE_2 = 8
NUM_EPOCHS = NUM_EPOCHS_OBSERVE + NUM_EPOCHS_TRAIN

class Brain(object):
    def __init__(self):
        print(device_lib.list_local_devices())
        self.model = Sequential()
        self.build_model()
        self.exp_memory = collections.deque(maxlen=MEMORY_SIZE) # Experience Replay Memory
        self.imp_exp_memory = collections.deque(maxlen=MEMORY_SIZE) # Supplementary Memory
        self.epsilon = INITIAL_EPSILON
        self.episode_count = 0

    def build_model(self):
        #self.model.add(Flatten(input_shape=(5)))
        self.model.add(Dense(24, activation="relu")) #1st Layer
        self.model.add(Dense(24, activation="relu")) #2nd Layer
        self.model.add(Dense(24, activation="relu")) #3rd Layer
        self.model.add(Dense(24, activation="relu")) #4th Layer
        self.model.add(Dense(24, activation="relu")) #5th Layer
        self.model.add(Dense(24, activation="relu")) #6th Layer
        self.model.add(Dense(3, activation="linear")) #last Layer
        self.model.build(input_shape=(None,5))
        self.model.compile(optimizer=Adam(learning_rate=LEARNING_RATE, clipnorm=1.), loss="mse")
        
        self.model.summary()
    
    def predict(self, s_t):
        # Return expected Q

        #Preprocess s_t data (Expand Dimension)
        s_t = np.expand_dims(s_t,0)
        
        ####DEBUG####
        #print(s_t)
        #print(np.shape(s_t))
        
        return self.model.predict(s_t)
    
    def think(self, s_t):
        if np.random.rand() < self.epsilon:
            # ε-greedy (random explorations)
            return np.random.randint(0,3), -100.0
        else:
            # Return action with the largest Q prediction
            a_t = self.predict(s_t)
            return np.argmax(a_t), np.max(a_t)
        
    def get_next_batch(self):
        # Create Batch
        batch_indices_1 = np.random.randint(low=0, high=len(self.exp_memory), size=BATCH_SIZE_1)
        #batch_indices_2 = np.random.randint(low=0, high=len(self.imp_exp_memory), size=BATCH_SIZE_2)
        batch_1 = [self.exp_memory[i] for i in batch_indices_1]
        #batch_2 = [self.imp_exp_memory[i] for i in batch_indices_2]
        batch = batch_1# + batch_2
        
        X = np.zeros((BATCH_SIZE, 5)) # input
        Y = np.zeros((BATCH_SIZE, 3)) # target(Q table)
        for i in range(BATCH_SIZE):
            s_t, a_t, r_t, s_tp1, game_over = batch[i]
            X[i] = s_t
            Y[i] = self.predict(s_t)

            # Update target Q with reward data
            if game_over: # Last frame: Q_t = r_t
                Y[i, a_t] = r_t
            else: # Not the last frame: Q_t = r_t + Q_t+1
                Q_sa = np.max(self.predict(s_tp1))
                Y[i, a_t] = r_t + GAMMA * Q_sa

        return X, Y

