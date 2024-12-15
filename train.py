import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten


DATASET_PATH = 'gesture_landmark_dataset'
GESTURES = ['Hand-close', 'Hand-open']

data = []
labels = []


for gesture in GESTURES:
    gesture_folder = os.path.join(DATASET_PATH, gesture)
    for csv_file in os.listdir(gesture_folder):
        file_path = os.path.join(gesture_folder, csv_file)
        landmarks = np.loadtxt(file_path, delimiter=',')  
        data.append(landmarks.flatten())  
        labels.append(gesture)


data = np.array(data)
labels = np.array(labels)


le = LabelEncoder()
labels = le.fit_transform(labels)


X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)


X_train = X_train.reshape(X_train.shape[0], 21, 3)  
X_test = X_test.reshape(X_test.shape[0], 21, 3)


model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(21, 3)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(len(GESTURES), activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save('models.h5')
