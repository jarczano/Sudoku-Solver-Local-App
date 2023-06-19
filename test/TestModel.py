import unittest
import os
import cv2
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np


def display_img(img):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    plt.show()


class TestModel(unittest.TestCase):
    def test_train_model(self):
        """
        Function test model digit recognition on real photo on sudoku square
        :return:
        """

        #path_read_image = r"..\test\Test_image"
        path_read_image = r"..\test\Test_number"
        directory = os.listdir(path_read_image)
        model = load_model(r"..\model\model1685451786.3455136.h5")
        #checkpoint_path = r"../model/cp-0020.h5"

        list_image = []
        for file in directory:
            list_image.append(file)

        counter_correct = 0
        for i in range(len(list_image)):
            img = cv2.imread(str(path_read_image) + r"/" + list_image[i], 0)
            img = cv2.resize(img, (28, 28))
            img = 255 - img
            img2 = img / 255
            _, img2 = cv2.threshold(img2, 150 / 255, 250 / 255, cv2.THRESH_BINARY)
            img2 = cv2.blur(img2, (2, 2))

            img2 = img2.reshape(1, 28, 28, 1)

            prob = np.max(model.predict(img2), axis=1)
            prediction = np.argmax(model.predict(img2), axis=1) + 1

            #print("{} with probability: {}".format(prediction, prob))
            #cv2.imshow('img', img)
            #cv2.waitKey(0)

            if prediction[0] == int(list_image[i][0]):
                counter_correct += 1

        print("Correct recognition: ", counter_correct, " / ", len(list_image))
        assert (counter_correct == len(list_image))


if __name__ == '__main__':
    unittest.main()
