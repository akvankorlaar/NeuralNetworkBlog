from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import tensorflow as tf

# Gather our data
df = pd.read_csv('preprocessed_shrub_dataset.csv')
# Shuffle our data
df = df.sample(frac=1)

# Turn our data into numpy arrays
X = df.iloc[:,1:3].values
y = df.iloc[:,3:4].values

# Define a neural network with 2 hidden layers
model = Sequential()
model.add(Dense(3, input_dim=2, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile our neural network
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train our neural network
history = model.fit(X, y, epochs=100, batch_size=5)
