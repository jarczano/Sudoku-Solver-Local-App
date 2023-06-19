import cv2
from scipy import ndimage
import heapq
import numpy as np


def find_sudoku_board(frame):
    '''
    Function looks for a sudoku board
    :param frame: video frame
    :return: if find sudoku board -> found=True and image of sudoku board else found=False and None
    '''

    height, width = frame.shape[0], frame.shape[1]

    approx_epsilon = 0.05 * width
    screen_area = width * height
    n_largest = 1
    found = False
    board = None

    # image processing
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    blur = cv2.medianBlur(blur, 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=5)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # checking the n_largest contour if the board sudoku is there
    largest_contours = heapq.nlargest(n_largest, contours, key=cv2.contourArea)

    # selecting such contour that: 1) has 4 vertices, 2) its not in the edge of image, 3) has suitable area,
    # 4) has convex shape, 5) is approximately a square
    for i in range(n_largest):
        approx_contour = cv2.approxPolyDP(largest_contours[i], approx_epsilon, True)
        if approx_contour.shape[0] == 4:
            if 0 not in approx_contour:
                if 0.37 * screen_area < cv2.contourArea(approx_contour) < 0.5 * screen_area:
                    if cv2.isContourConvex(approx_contour):
                        if 0.95 * cv2.contourArea(approx_contour) < (cv2.arcLength(approx_contour, True) / 4) ** 2 < 1.05 * cv2.contourArea(approx_contour):
                            found = True
                            break

    def protractor(contour):
        # return the angle between top horizontal line of contour and axis ox
        if len(contour) == 4:
            contour = np.amin(contour, axis=1)
            contour = contour[contour[:, 1].argsort()]

            # P1 is point with smaller x value, P2 is point with grater x value
            if contour[0][0] < contour[1][0]:
                x1, y1 = contour[0][0], contour[0][1]
                x2, y2 = contour[1][0], contour[1][1]
            else:
                x1, y1 = contour[1][0], contour[1][1]
                x2, y2 = contour[0][0], contour[0][1]

            if y1 > y2:
                angle = np.arctan((y1 - y2) / (x2 - x1)) * 180 / np.pi
                clockwise = True
            else:
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                clockwise = False

            return (-2 * clockwise + 1) * angle

    if found:
        angle = protractor(approx_contour)
        x, y, w, h = cv2.boundingRect(approx_contour)
        board = img[y:y+h, x:x+w]
        board = ndimage.rotate(board, angle, cval=255)

    return found, board


# to zas wyjebac
def find_sudoku_board_2(frame): # do zrobienia zdjec plansz
    '''
    Function looks for a sudoku board
    :param frame: video frame
    :return: if find sudoku board -> found=True and image of sudoku board else found=False and None
    '''

    height, width = frame.shape[0], frame.shape[1]

    approx_epsilon = 0.05 * width
    screen_area = width * height
    n_largest = 1
    found = False
    board = None

    # image processing
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (7, 7), 0)
    blur = cv2.medianBlur(blur, 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    thresh = cv2.dilate(thresh, kernel, iterations=5)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # checking the n_largest contour if the board sudoku is there
    largest_contours = heapq.nlargest(n_largest, contours, key=cv2.contourArea)

    # selecting such contour that: 1) has 4 vertices, 2) its not in the edge of image, 3) has suitable area,
    # 4) has convex shape, 5) is approximately a square
    for i in range(n_largest):
        approx_contour = cv2.approxPolyDP(largest_contours[i], approx_epsilon, True)
        if approx_contour.shape[0] == 4:
            if 0 not in approx_contour:
                if 0.37 * screen_area < cv2.contourArea(approx_contour) < 0.5 * screen_area:
                    if cv2.isContourConvex(approx_contour):
                        if 0.95 * cv2.contourArea(approx_contour) < (cv2.arcLength(approx_contour, True) / 4) ** 2 < 1.05 * cv2.contourArea(approx_contour):
                            found = True
                            break

    def protractor(contour):
        # return the angle between top horizontal line of contour and axis ox
        if len(contour) == 4:
            contour = np.amin(contour, axis=1)
            contour = contour[contour[:, 1].argsort()]

            # P1 is point with smaller x value, P2 is point with grater x value
            if contour[0][0] < contour[1][0]:
                x1, y1 = contour[0][0], contour[0][1]
                x2, y2 = contour[1][0], contour[1][1]
            else:
                x1, y1 = contour[1][0], contour[1][1]
                x2, y2 = contour[0][0], contour[0][1]

            if y1 > y2:
                angle = np.arctan((y1 - y2) / (x2 - x1)) * 180 / np.pi
                clockwise = True
            else:
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                clockwise = False

            return (-2 * clockwise + 1) * angle

    if found:
        path_to_save = r"catch_board3.jpg"
        print(found)
        cv2.imwrite(path_to_save, frame)
        cv2.imshow('img', frame)
        cv2.waitKey(0)
        #angle = protractor(approx_contour)
        #x, y, w, h = cv2.boundingRect(approx_contour)
        #board = img[y:y+h, x:x+w]
        #board = ndimage.rotate(board, angle, cval=255)

    return found, board