from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
import keras

#Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Reshape images (flatten into a single row)
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0], -1)

#Number of classes, number of features
num_classes = 10
num_features = x_train.shape[-1]

#Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Normalize
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#Model
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=num_features))
model.add(Dense(units=32, activation='relu', input_dim=num_features))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

#Fit
model.fit(x_train, y_train, epochs=10, batch_size=32)

#Evaluation
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
print("Accuracy:", loss_and_metrics[1])

#Predictions
classes = model.predict(x_test, batch_size=128)
