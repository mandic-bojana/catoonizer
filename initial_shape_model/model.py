from keras import Sequential
from keras.layers import Dense, Conv2D, PReLU, Dropout, BatchNormalization, MaxPool2D, Dropout, InputLayer, Input, Flatten


def create_model():

    model = Sequential()

# I
    model.add(Conv2D(24, kernel_size=(11, 11), strides=(4, 4), input_shape=(320, 240, 3)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3, 2))

# II
    model.add(Conv2D(64, kernel_size=(5, 5), strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPool2D(3, 2))

# III
    model.add(Conv2D(196, kernel_size=(3, 3), strides=(1, 1)))
    model.add(PReLU())

# IV
    model.add(Conv2D(96, kernel_size=(3, 3), strides=(1, 1)))
    model.add(PReLU())

# flat
    model.add(Flatten())

# V
    model.add(Dense(1024))
    model.add(PReLU())
    model.add(Dropout(0.2))

# VI
    model.add(Dense(1024))
    model.add(PReLU())
    model.add(Dropout(0.25))

# VII
    model.add(Dense(136))

    return model
