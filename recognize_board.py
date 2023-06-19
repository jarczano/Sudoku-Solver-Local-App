from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt


def recognize_board(board, sorted_split_board):
    """
    :param board: image of sudoku board
    :param sorted_split_board: list of sorting contour squares from top left to bottom right
    :return: digital sudoku board as numpy array
    """

    model = load_model(r'model\model1685400012.8494081.h5')
    digital_board = np.zeros((9, 9), dtype=np.uint8)
    counter = 324
    # tutaj lepiej by bylo najpierw tresholdy i blury dla calego zdjecia
    for i in range(9):
        for j in range(9):
            counter += 1
            x, y, w, h = cv2.boundingRect(sorted_split_board[i][j])
            margin_top_bottom, margin_left_right = int(0.1 * h), int(0.1 * w)
            #one_square = board[y + margin_top_bottom: y + h - margin_top_bottom, x + margin_left_right: x + w - margin_left_right] # to bardziej sie nadaje tylko dla niskiej rozdzielczosci
            one_square = board[y: y + h, x: x + w]



            # test wyświetlania na bierząco kwadratow
            #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            #cv2.imshow('img', one_square)
            #cv2.waitKey(0)

            # to zbierania obrazow malych kwadratow
            one_square = cv2.resize(one_square, (28, 28))
            #path_write = r'neural_network\data\dataset\noise'
            #cv2.imwrite(str(path_write) + r'\{}.jpg'.format(counter), one_square)


            one_square = 255 - one_square
            one_square = one_square / 255
            _, one_square = cv2.threshold(one_square, 150/255, 250/255, cv2.THRESH_BINARY)
            one_square = cv2.blur(one_square, (2, 2)) # mozew []



            one_square = one_square.reshape(1, 28, 28, 1)


            # if model predict digit 0-9 with probability >40% then accepted, [10] means empty square

            prediction = np.argmax(model.predict(one_square), axis=1)
            max_probability = model.predict(one_square)[0][prediction[0]]
            if max_probability > 0.4 and prediction != [10]:
                digital_board[i][j] = prediction[0]


            # testy
            print("row: ", i, " col: ", j, "prediction: ",prediction, 'probability: ', max_probability)


            #if i == 3 and j == 8:
            #fig = plt.figure(figsize=(12, 10))
            #ax = fig.add_subplot(111)
            #ax.imshow(one_square.reshape(28, 28), cmap='gray')
            #plt.show()

    return digital_board

def recognize_board2(board, sorted_split_board):
    """
    :param board: image of sudoku board
    :param sorted_split_board: list of sorting contour squares from top left to bottom right
    :return: digital sudoku board as numpy array
    """

    model_binary = load_model(r'model\model_binary1685438852.1508806.h5')
    model = load_model(r'model\model1685451786.3455136.h5')
    digital_board = np.zeros((9, 9), dtype=np.uint8)
    counter = 324
    # tutaj lepiej by bylo najpierw tresholdy i blury dla calego zdjecia
    for i in range(9):
        for j in range(9):
            counter += 1
            x, y, w, h = cv2.boundingRect(sorted_split_board[i][j])
            margin_top_bottom, margin_left_right = int(0.1 * h), int(0.1 * w)
            one_square = board[y + margin_top_bottom: y + h - margin_top_bottom, x + margin_left_right: x + w - margin_left_right] # to bardziej sie nadaje tylko dla niskiej rozdzielczosci
            #one_square = board[y: y + h, x: x + w]



            # test wyświetlania na bierząco kwadratow
            #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            #cv2.imshow('img', one_square)
            #cv2.waitKey(0)

            # to zbierania obrazow malych kwadratow
            one_square = cv2.resize(one_square, (28, 28))
            #path_write = r'neural_network\data\dataset\noise'
            #cv2.imwrite(str(path_write) + r'\{}.jpg'.format(counter), one_square)


            one_square = 255 - one_square
            one_square = one_square / 255
            _, one_square = cv2.threshold(one_square, 150/255, 250/255, cv2.THRESH_BINARY)
            one_square = cv2.blur(one_square, (2, 2)) # mozew []



            one_square = one_square.reshape(1, 28, 28, 1)


            # if model predict digit 0-9 with probability >40% then accepted, [10] means empty square

            # if empty or digit
            prediction_binary = np.argmax(model_binary.predict(one_square), axis=1)
            if prediction_binary[0] == 0:
                # tutaj 2 razy robi prediction
                prediction = np.argmax(model.predict(one_square), axis=1)[0] + 1
                #max_probability = model.predict(one_square)[0][prediction]
                #if max_probability > 0.4:
                digital_board[i][j] = prediction
                #print("Prediction: ", prediction)

            # testy
                #print("row: ", i, " col: ", j, "prediction: ",prediction, 'probability: ', max_probability)


            #if i == 3 and j == 8:
            #fig = plt.figure(figsize=(12, 10))
            #ax = fig.add_subplot(111)
            #ax.imshow(one_square.reshape(28, 28), cmap='gray')
            #plt.show()

    return digital_board