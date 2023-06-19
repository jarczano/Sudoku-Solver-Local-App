import os
import numpy as np
import cv2
from data_generator import generate_data_digits, generate_data_binary


def load_dataset(data):
    """
     The function loads a dataset. If the dataset has not been generated previously, it will try create it first.
    :param data: indicate which data should be loaded: "binary" or "digits"
    :return: if success=True: dataset of digit as numpy array
    """

    if data == "binary":
        path_read_image = r"data\dataset_binary\images"
        path_read_data = r"data\dataset_binary\numpy file"
    elif data == "digits":
        path_read_image = r"data\dataset_digits\images"
        path_read_data = r"data\dataset_digits\numpy file"

    directory = os.listdir(path_read_data)
    X = np.empty((0, 28, 28), dtype=np.uint8)

    data_files = []
    for file in directory:
        if file.lower().startswith('data'):
            data_files.append(file)

    # Check if .npy dataset exist
    success = True
    if len(data_files) == 0:
        #  Check if image dataset exist
        if len(os.listdir(path_read_image)) < 10:
            # Try generate it
            if data == "binary":
                success = generate_data_binary()
            elif data == "digits":
                success = generate_data_digits()
            if success:
                image_data_2_npy(data=data)
        else:
            image_data_2_npy(data=data)

    if success:
        files = []
        for i in range(len(data_files)):
            files.append(str(path_read_data) + r'\{}'.format(data_files[i]))

        for file in files:
            arr = np.load(file)
            X = np.concatenate((X, arr), axis=0)
    return success, X


def load_label(data):
    """
    The function loads a label for dataset.
    :param data: indicate which data label should be loaded: "binary" or "digits"
    :return: Labels for Dataset of digit as numpy array
    """

    if data == "binary":
        path_read_label = r"data\dataset_binary\numpy file"
    elif data == "digits":
        path_read_label = r"data\dataset_digits\numpy file"
    directory = os.listdir(path_read_label)
    Y = np.empty(0)  # czy uint robi roznice

    label_file = ''
    for file in directory:
        if file.lower().startswith('label'):
            label_file = file
    arr = np.load(str(path_read_label) + '\{}'.format(label_file))
    Y = np.concatenate((Y, arr), axis=0)
    return Y


def image_data_2_npy(data, batch_size=10000):
    """
    Function takes all data images and converts them into numpy files with batch division.
    :param batch_size: number of image in one batch
    :param data: which data converts to numpy files: "digits" or "binary"
    :return: Save dataset images as .npy file
    """

    if data == "digits":
        path_read = r"data\dataset_digits\images"
        path_write = r"data\dataset_digits\numpy file"
    elif data == "binary":
        path_read = r"data\dataset_binary\images"
        path_write = r"data\dataset_binary\numpy file"

    files = os.listdir(path_read)
    num_files = len(files)

    whole_num_batches = num_files // batch_size
    rest_num_files = num_files % batch_size

    for batch in range(whole_num_batches):
        X = np.ndarray((batch_size, 28, 28), dtype=np.uint8)
        batch_factor = batch * batch_size
        for image in range(batch_size):
            image_name = str(path_read) + r'\{}.jpg'.format(batch_factor + image)
            img = cv2.imread(image_name, 0)
            img = 255 - img
            X[image::] = img

        np.save(str(path_write) + r"\data{}.npy".format(batch), X)

    if rest_num_files != 0:
        X = np.ndarray((rest_num_files, 28, 28), dtype=np.uint8)
        for image in range(rest_num_files):
            image_name = str(path_read) + r"\{}.jpg".format(whole_num_batches * batch_size + image)
            img = cv2.imread(image_name, 0)
            img = 255 - img
            X[image::] = img
        np.save(str(path_write) + r"\data{}.npy".format(whole_num_batches), X)
