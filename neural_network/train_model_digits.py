import time
import tensorflow
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.models import load_model
from sklearn.model_selection import train_test_split
from utils import load_label, load_dataset


def train_model_digits():
    """ Digits 1 - 9
    Function trains a digit recognition model
    :return: Save model.h5 for digit recognition and information about his architecture and summary as model.txt
    """
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    physical_devices = tensorflow.config.list_physical_devices("GPU")
    if len(physical_devices) >= 1:
        tensorflow.config.experimental.set_memory_growth(physical_devices[0], True)

    # Load dataset and label
    success, X = load_dataset(data='digits')
    if success:
        Y = load_label('digits')

        # Split datasets
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

        # One hot encoding
        y_cat_test = to_categorical(y_test - 1, 9)
        y_cat_train = to_categorical(y_train - 1, 9)

        x_train = x_train / 255
        x_test = x_test / 255

        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

        # Build model

        path_save_model = r'../model'

        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (5, 5), activation='relu', padding='same', strides=2))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(9, activation='softmax'))


        optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Continue learning
        #checkpoint_path = r"../model/model1685379747.0931883.h5"
        #model = tensorflow.keras.models.load_model(checkpoint_path)

        checkpoint_path = r"../model/cpr-{epoch:04d}.h5"
        cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False, verbose=1, period=5)

        history = model.fit(x_train, y_cat_train, epochs=2, validation_data=(x_test, y_cat_test), callbacks=[cp_callback])
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.plot(epochs, loss, 'bo', label='Loss train')
        plt.plot(epochs, val_loss, 'b', label='Loss validation')
        plt.title('Loss train and validation ')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        plt.plot(epochs, acc, 'bo', label='Accuracy train')
        plt.plot(epochs, val_acc, 'b', label='Accuracy validation')
        plt.title('Accuracy train and validation')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

        model_name = "model{}".format(time.time())
        model.save(str(path_save_model) + r"/{}.h5".format(model_name))

        # Create information file about model
        with open(path_save_model + r"/{}.txt".format(model_name), 'w') as file:
            file.write("Architecture model: \n")
            model.summary(print_fn=lambda x: file.write(x + '\n'))

            file.write("\nResult training:\n")
            file.write("Accuracy: {}\n".format(history.history['accuracy']))
            file.write("Loss: {}\n".format(history.history['loss']))
            file.write("Validation accuracy): {}\n".format(history.history['val_accuracy']))
            file.write("Validation loss): {}\n".format(history.history['val_loss']))
    else:
        print(r"No valid fonts. Please place the font files .ttf in the Sudoku Solver\neural_network\data\source_data folder. "
              r"Fonts can be downloaded, for example, from https://fonts.google.com/")


if __name__ == '__main__':
    train_model_digits()
