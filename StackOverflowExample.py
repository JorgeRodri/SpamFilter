import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

model = Sequential()
model.add(Dense(32, input_dim=500))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

import numpy as np
from sklearn.model_selection import train_test_split
X = np.random.random((1200, 500))
y = np.random.randint(2, size=(1200, 1))

X_train, X_test, y_train, y_test = train_test_split(X, keras.utils.to_categorical(y, num_classes=2), test_size=1/3)

tensorboard = keras.callbacks.TensorBoard(log_dir="TFlogs",
                                          histogram_freq=2, batch_size=300,
                                          write_grads=True, write_images=True)

history = model.fit(X_train, y_train, epochs=10,
                    validation_split=0.1, batch_size=300, callbacks=[tensorboard])