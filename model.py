import numpy as np
from sklearn.model_selection import train_test_split
from keras import regularizers
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.optimizers import SGD, Adam, RMSprop

data = np.load('D:/Dataset/Data/data_Train.npy')
label = np.load('D:/Dataset/Data/label_Train.npy')

train_data_X = np.array([i for i in data]).reshape(-1,30,30,3)
train_data_Y = np.array([i for i in label]).reshape(-1,1)

print(train_data_X.shape)
print(train_data_Y.shape)

X_train, X_test, y_train, y_test = train_test_split(train_data_X, train_data_Y, test_size=0.2, random_state=24)

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

##model = Sequential()
##model.add(Conv2D(32, (3, 3), input_shape=X_train.shape[1:], activation='relu'))
##model.add(Conv2D(32, (3, 3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Conv2D(64, (3, 3), activation='relu'))
##model.add(Conv2D(64, (3, 3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Conv2D(128, (3, 3), activation='relu'))
##model.add(Conv2D(128, (3, 3), activation='relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##model.add(Flatten())
##model.add(Dense(256, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(256, activation='relu'))
##model.add(Dropout(0.5))
##model.add(Dense(43, activation='softmax'))
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

model_json = model.to_json()
open('traffic_sign.json', 'w').write(model_json)
model.save_weights('traffic_sign.h5', overwrite=True)
