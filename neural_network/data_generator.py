from PIL import ImageFont, ImageDraw, Image
import os
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np


def generate_data_binary(number_augmentation_digit=10, number_augmentation_noise=10):
    """
    Function generate dataset of images of full fields with digits and empty fields with noise.
    Data augmentation is used to increase the dataset.
    In addition, the function creates a file of labels
    :param number_augmentation_digit: Number of augmentations for each font
    :param number_augmentation_noise:Number of augmentations for each noise image
    :return: success=True if creates dataset of images, success=False if not creates dataset of images
    """

    path_read = r'data/source_data/Fonts'
    path_read_noise = r'data/source_data/noise/'
    path_write = r'data/dataset_binary/images'
    path_write_label = r"data/dataset_binary/numpy file"

    image_data_generator = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,    # randomly flip images
            fill_mode='nearest',
            brightness_range=[0.2, 1.7])

    # Digit dataset generate
    directory = os.listdir(path_read)
    fonts_name = []
    for file in directory:
        if file.lower().endswith('.ttf'):
            fonts_name.append(file)

    if len(fonts_name) < 1:
        return False
    else:
        counter = 0
        digits = [i for i in range(10)]

        for digit in digits:
            for font in fonts_name:

                # PIL draw font
                image = Image.open(r'data\source_data\support_images\blank.jpg')
                draw = ImageDraw.Draw(image)
                text_font = ImageFont.truetype(str(path_read)+r'\{}'.format(font), 20)
                draw.text((5, 0), text=str(digit), font=text_font)

                img_number = np.asarray(image)
                img_number = img_number.reshape(1, 28, 28, 1)
                aug_iter = image_data_generator.flow(img_number)

                for i in range(number_augmentation_digit):
                    image_aug = next(aug_iter)[0]
                    cv2.imwrite(str(path_write)+r'\{}.jpg'.format(counter), image_aug)
                    counter += 1

        # Noise dataset generate
        directory_noise = os.listdir(path_read_noise)

        noise_image_name = []
        for file in directory_noise:
                noise_image_name.append(file)

        for image_name in noise_image_name:
            image = Image.open(str(path_read_noise) + image_name)
            img_number = np.asarray(image)
            img_number = img_number.reshape(1, 28, 28, 1)
            aug_iter = image_data_generator.flow(img_number)

            for i in range(number_augmentation_noise):
                image_aug = next(aug_iter)[0]
                cv2.imwrite(str(path_write) + r'\{}.jpg'.format(counter), image_aug)
                counter += 1

        # Create label dataset
        Y = []
        for digit in digits:
            Y.extend([0] * (number_augmentation_digit * len(fonts_name)))
        Y.extend([1] * (number_augmentation_noise * len(directory_noise)))
        Y = np.asarray(Y, dtype=np.uint8)
        np.save(str(path_write_label) + r"\Label.npy", Y)
        return True


def generate_data_digits(number_augmentation_digit=10):
    """
    Function generate dataset of images of digits from 1 to 9 based on a set of various fonts.
    Data augmentation is used to increase the dataset.
    In addition, the function creates a file of labels.
    :param number_augmentation_digit: Number of augmentations for each font
    :return: success=True if creates dataset of images, success=False if not creates dataset of images
    """

    path_read = r'data\source_data\Fonts'
    path_write = r'data\dataset_digits\images'
    path_write_label = r"data\dataset_digits\numpy file"

    image_data_generator = ImageDataGenerator(
            featurewise_center=True,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.1,  # Randomly zoom image
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,    # randomly flip images
            fill_mode='nearest',
            brightness_range=[0.2, 1.7])

    # Digit dataset generate
    directory = os.listdir(path_read)
    fonts_name = []
    for file in directory:
        if file.lower().endswith('.ttf'):
            fonts_name.append(file)

    if len(fonts_name) < 1:
        return False
    else:

        counter = 0
        digits = [i for i in range(1, 10)]

        for digit in digits:
            for font in fonts_name:
                # PIL draw font

                image = Image.open(r'data\source_data\support_images\blank.jpg')
                draw = ImageDraw.Draw(image)
                text_font = ImageFont.truetype(str(path_read)+r'\{}'.format(font), 20)
                draw.text((5, 0), text=str(digit), font=text_font)

                img_number = np.asarray(image)
                img_number = img_number.reshape(1, 28, 28, 1)
                aug_iter = image_data_generator.flow(img_number)

                for i in range(number_augmentation_digit):
                    image_aug = next(aug_iter)[0]
                    cv2.imwrite(str(path_write)+r'\{}.jpg'.format(counter), image_aug)
                    counter += 1

        # Create label dataset
        Y = []
        for digit in digits:
            Y.extend([digit] * (number_augmentation_digit * len(fonts_name)))
        Y = np.asarray(Y, dtype=np.uint8)
        np.save(str(path_write_label) + r"\Label.npy", Y)
        return True




