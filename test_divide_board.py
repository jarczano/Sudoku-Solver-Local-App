import math

import cv2
import os
import numpy as np
import heapq
from scipy import ndimage


def euclidian_distance(point_a, point_b):
    return math.sqrt((point_b[0] - point_a[0]) ** 2 + (point_b[1] - point_a[1]) ** 2)

def big_contour(frame):

    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # Ustawienie normalnego trybu okna

    height, width = frame.shape[0], frame.shape[1]

    approx_epsilon = 0.05 * width
    screen_area = width * height
    n_largest = 5
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

    #cv2.imshow("frame", thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contours = heapq.nlargest(n_largest, contours, key=cv2.contourArea)

    # selecting such contour that: 1) has 4 vertices, 2) its not in the edge of image, 3) has suitable area,
    # 4) has convex shape, 5) is approximately a square
    for i in range(n_largest):
        approx_contour = cv2.approxPolyDP(largest_contours[i], approx_epsilon, True)
        if approx_contour.shape[0] == 4:
            if 0 not in approx_contour:
                if 0.37 * screen_area < cv2.contourArea(approx_contour) < 0.7 * screen_area:
                    if cv2.isContourConvex(approx_contour):

                        cv2.drawContours(frame, approx_contour, -1, (0, 255, 0), 3)
                        cv2.imshow("frame", frame)

                        # check if contour is square
                        contour = np.amin(approx_contour, axis=1)
                        contour = contour[contour[:, 1].argsort()]
                        top = contour[0:2]
                        bottom = contour[2:4]
                        top = top[top[:, 0].argsort()]
                        bottom = bottom[bottom[:, 0].argsort()]
                        top_left = top[0]
                        top_right = top[1]
                        bottom_left = bottom[0]
                        bottom_right = bottom[1]
                        top_length = euclidian_distance(top_left, top_right)
                        left_side_length = euclidian_distance(top_left, bottom_left)
                        right_side_length = euclidian_distance(top_right, bottom_right)
                        bottom_length = euclidian_distance(bottom_right, bottom_left)

                        #if 0.9 * top_length< left_side_length & right_side_length & bottom_length < 1.1 * top_length:
                        if all(0.95 * top_length < val < 1.05 * top_length for val in (top_length, left_side_length, right_side_length, bottom_length)):

                        #pole = cv2.contourArea(approx_contour)
                        #bok = cv2.arcLength(approx_contour, True) / 4
                        #if 0.95 * cv2.contourArea(approx_contour) < (cv2.arcLength(approx_contour, True) / 4) ** 2 < 1.05 * cv2.contourArea(approx_contour):
                            found = True
                            gib_c = [approx_contour]

                            # test łapania całej planszy
                            cv2.drawContours(frame, gib_c, -1, (0, 255, 0), 3)
                            cv2.imshow("frame", frame)
                            cv2.waitKey(0)
                            break
    if found == False:
        gib_c = None
    return found, gib_c


def big_contour2(frame):
    """
    F return contour of board square
    :param frame:
    :return:
    """
    #cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # Ustawienie normalnego trybu okna

    height, width = frame.shape[0], frame.shape[1]



    n_largest = 5
    found = False
    board = None


    # resize image
    width_resize = 640
    height_resize = int(width_resize * height / width)
    screen_area = width_resize * height_resize

    scale = width / width_resize
    approx_epsilon = 0.05 * width_resize

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (width_resize, height_resize))

    image = cv2.medianBlur(image, 3)
    edges = cv2.Canny(image, 50, 150)


    #cv2.imshow("frame", edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contours = heapq.nlargest(n_largest, contours, key=cv2.contourArea)

    # selecting such contour that: 1) has 4 vertices, 2) its not in the edge of image, 3) has suitable area,
    # 4) has convex shape, 5) is approximately a square
    for i in range(n_largest):
        approx_contour = cv2.approxPolyDP(largest_contours[i], approx_epsilon, True)
        if approx_contour.shape[0] == 4:
            if 0 not in approx_contour:
                if 0.37 * screen_area < cv2.contourArea(approx_contour) < 0.7 * screen_area:# 0.37<
                    if cv2.isContourConvex(approx_contour):

                        #cv2.drawContours(frame, [approx_contour], -1, (0, 255, 0), 3)
                        #cv2.imshow("frame", frame)

                        # check if contour is square
                        contour = np.amin(approx_contour, axis=1)
                        contour = contour[contour[:, 1].argsort()]
                        top = contour[0:2]
                        bottom = contour[2:4]
                        top = top[top[:, 0].argsort()]
                        bottom = bottom[bottom[:, 0].argsort()]
                        top_left = top[0]
                        top_right = top[1]
                        bottom_left = bottom[0]
                        bottom_right = bottom[1]
                        top_length = euclidian_distance(top_left, top_right)
                        left_side_length = euclidian_distance(top_left, bottom_left)
                        right_side_length = euclidian_distance(top_right, bottom_right)
                        bottom_length = euclidian_distance(bottom_right, bottom_left)

                        if all(0.95 * top_length < val < 1.05 * top_length for val in (top_length, left_side_length, right_side_length, bottom_length)):

                            found = True
                            gib_c = scale * approx_contour
                            gib_c = gib_c.astype(np.int32)
                            #my_contour = my_contour.astype(np.int32)
                            #gib_c = approx_contour


                            # lapanie planszy resize
                            #resize_approx_contour = scale * approx_contour
                            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
                            cv2.drawContours(frame, [gib_c], -1, (0, 255, 0), 5)

                            cv2.imshow("frame", frame)
                            #cv2.waitKey(0)

                            # test łapania całej planszy w normalnym rozmiarz
                            #cv2.drawContours(frame, gib_c, -1, (0, 255, 0), 3)
                            #cv2.imshow("frame", frame)
                            #cv2.waitKey(0)
                            break
    if found == False:
        gib_c = None
    return found, gib_c





def t():

    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))
    contours = big_contour(image)
    print(contours)

    #cv2.drawContours(image, contours, -1, (0, 255, 0), 3, cv2.CHAIN_APPROX_NONE)
    cv2.polylines(image, contours, True, (0, 255, 0), 2)
    #print(contours)


    cv2.namedWindow('img', cv2.WINDOW_NORMAL)  # Ustawienie normalnego trybu okna

    cv2.imshow('img', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#t()

#contour = np.array([[[900,  850]], [[76,  774]], [[877, 47]], [[10, 15]]])
#contour = np.array([[[900,  850]], [[76,  774]], [[877, 15]], [[10, 47]]]) # 4, 3 ,2,1
def test():
    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))
    contour = big_contour(image)

    contour = np.amin(contour, axis=1)
    contour = contour[contour[:, 1].argsort()]
    TOP = contour[0:2]
    BOTTOM = contour[2:4]
    TOP = TOP[TOP[:, 0].argsort()]
    BOTTOM = BOTTOM[BOTTOM[:, 0].argsort()]
    print("top left:", TOP[0])
    print("top right:", TOP[1])
    print("bottom left:", BOTTOM[0])
    print("bottom right:", BOTTOM[1])
    top_left = TOP[0]
    top_right = TOP[1]
    bottom_left = BOTTOM[0]
    bottom_right = BOTTOM[1]

    top_partition = [top_left]
    left_partition = [top_left]
    right_partition = [top_right]
    bottom_partition = [bottom_left]
    Xtop = np.empty((2,1))

    for part in range(1, 10):
        #arr = top_left + (top_right - top_left) * part / 9
        #Xtop = np.concatenate((Xtop, arr), axis=1)
        #x = top_left + (top_right - top_left) * part / 9
        top_partition.append(top_left + (top_right - top_left) * part / 9)
        left_partition.append(top_left + (bottom_left - top_left) * part / 9)
        right_partition.append(top_right + (bottom_right - top_right) * part / 9)
        bottom_partition.append(bottom_left + (bottom_right - bottom_left) * part / 9)

        #print(TOP_LEFT + (TOP_RIGHT - TOP_LEFT) * part / 9)


    point_matrix = np.zeros(shape=(10, 10, 2))
    for row in range(1, 9):
        for col in range(1, 9):
            alfa = (bottom_partition[col][1] - top_partition[col][1]) / (bottom_partition[col][0] - top_partition[col][0])
            beta = top_partition[col][1] - (bottom_partition[col][1] - top_partition[col][1]) / (bottom_partition[col][0] - top_partition[col][0]) * top_partition[col][0]
            gamma = (right_partition[row][1] - left_partition[row][1]) / (right_partition[row][0] - left_partition[row][0])
            delta = left_partition[row][1] - (right_partition[row][1] - left_partition[row][1]) / (right_partition[row][0] - left_partition[row][0]) * left_partition[row][0]

            x_coor = (delta - beta) / (alfa - gamma)
            y_coor = alfa * (delta - beta) / (alfa - gamma) + beta

            corner = np.array([x_coor, y_coor])
            point_matrix[row, col] = corner

    point_matrix[0][::] = np.array(top_partition)
    point_matrix[9][::] = np.array(bottom_partition)
    #point_matrix[::][0] = np.array(left_partition)
    #point_matrix[::][9] = np.array(right_partition)

    for row in range(10):
        point_matrix[row][0] = np.array(left_partition)[row]
        point_matrix[row][9] = np.array(right_partition)[row]

    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))
    #for row in range(10):
        #col = 0
        #print(point_matrix[row][col])
    for row in range(10):
        for col in range(10):
            #col = 1
            #print(point_matrix[row][col][1])
            x_center = point_matrix[row][col][0]
            y_center = point_matrix[row][col][1]
            #print(x_center)
            #print(y_center)
            cv2.circle(image, (int(x_center), int(y_center)), 3, (255, 0, 0), -1)

    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    contours_sq = []
    for row in range(9):
        for col in range(9):
            #xa = point_matrix[row][col]
            contours_sq.append([point_matrix[row][col], point_matrix[row + 1][col], point_matrix[row][col + 1], point_matrix[row + 1][col + 1]])

    contours_sq = np.array(contours_sq)
    print(contours_sq.shape)
    print(contours_sq[0])
    test_shape = contours_sq[0].reshape(4, 1, 2)
    counturs_sq_reshape = []
    for i in range(81):
        counturs_sq_reshape.append(contours_sq[i].reshape(4,1,2))






    contour_proper = np.array([[[900,  850]], [[76,  774]], [[877, 47]], [[10, 15]]])
    my_contour = counturs_sq_reshape[3]
    my_contour = my_contour.astype(np.int32)
    #test_con = np.array([[[10,  10]], [[100,  12]], [[12, 80]], [[100, 80]]])

    x, y, w, h = cv2.boundingRect(my_contour)
    one_square = image[y: y + h, x: x + w]
    cv2.imshow('img', one_square)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#test()
#one_square = image[y: y + h, x: x + w]

#print(contours_sq.shape)
#print(point_matrix)
#print(top_partition)
#print(top_partition[0])
#print('bla')
'''
TOP_partition_X = [TOP[0][0]]
TOP_partition_Y = [TOP[0][1]]
BOTTOM_partition_X = [BOTTOM[0][0]]
BOTTOM_partition_Y = [BOTTOM[0][1]]
LEFT_partition_X = [TOP[0][0]]
LEFT_partition_Y = [TOP[0][1]] # piersza wsp który kwadrat lewy czy prawy a druga wsp ktora wspolrzedna x czy y
RIGHT_partition_X = [TOP[1][0]] # [LEFT / RIGHT][X/Y]
RIGHT_partition_Y = [TOP[1][1]]
for part in range(1, 10):
    TOP_partition_X.append(TOP[0][0] + (TOP[1][0] - TOP[0][0]) * part / 9)
    TOP_partition_Y.append(TOP[0][1] + (TOP[1][1] - TOP[0][1]) * part / 9)

    BOTTOM_partition_X.append(BOTTOM[0][0] + (BOTTOM[1][0] - BOTTOM[0][0]) * part / 9)
    BOTTOM_partition_Y.append(BOTTOM[0][1] + (BOTTOM[1][1] - BOTTOM[0][1]) * part / 9)

    LEFT_partition_X.append(TOP[0][0] + (BOTTOM[0][0] - TOP[0][0]) * part / 9)
    LEFT_partition_Y.append(TOP[0][1] + (BOTTOM[0][1] - TOP[0][1]) * part / 9)

    RIGHT_partition_X.append(TOP[0][0] + (BOTTOM[1][0] - TOP[0][0]) * part / 9)
    RIGHT_partition_Y.append(TOP[1][1] + (BOTTOM[1][1] - TOP[1][1]) * part / 9)
    
    
print("top x: ", TOP_partition_X)
print("top y: ", TOP_partition_Y)
print("bottom x: ", BOTTOM_partition_X)
print("bottom y: ", BOTTOM_partition_Y)
print("left x: ", LEFT_partition_X)
print("left Y: ", LEFT_partition_Y)
print("right x: ", RIGHT_partition_X)
print("right Y: ", RIGHT_partition_Y)
print('-------------------------------------')
'''
'''
# corner
top_left_x = TOP[0][0]
top_left_y = TOP[0][1]

top_right_x = TOP[1][0]
top_right_y = TOP[1][1]

bottom_left_x = BOTTOM[0][0]
bottom_left_y = BOTTOM[0][1]

bottom_right_x = BOTTOM[1][0]
bottom_right_y = BOTTOM[1][1]
#-------------------
top_partition_x = [top_left_x]
top_partition_y = [top_left_y]

left_partition_x = [top_left_x]
left_partition_y = [top_left_y]

right_partition_x = [top_right_x]
right_partition_y = [top_right_y]

bottom_partition_x = [bottom_left_x]
bottom_partition_y = [bottom_left_y]

for part in range(1, 10):
    top_partition_x.append(top_left_x + (top_right_x - top_left_x) * part / 9)
    top_partition_y.append(top_left_y + (top_right_y - top_left_y) * part / 9)

    left_partition_x.append(top_left_x + (bottom_left_x - top_left_x) * part / 9)
    left_partition_y.append(top_left_y + (bottom_left_y - top_left_y) * part / 9)

    right_partition_x.append(top_right_x + (bottom_right_x - top_right_x) * part / 9)
    right_partition_y.append(top_right_y + (bottom_right_y - top_right_y) * part / 9)

    bottom_partition_x.append(bottom_left_x + (bottom_right_x - bottom_left_x) * part / 9)
    bottom_partition_y.append(bottom_left_y + (bottom_right_y - bottom_left_y) * part / 9)



print("top x: ", top_partition_x)
print("top y: ", top_partition_y)
print("bottom x: ", bottom_partition_x)
print("bottom y: ", bottom_partition_y)
print("left x: ", left_partition_x)
print("left Y: ", left_partition_y)
print("right x: ", right_partition_x)
print("right Y: ", right_partition_y)
print('=-----------')
print("top", top_partition)
print("bottom", bottom_partition)
print("left", left_partition)
print("right", right_partition)

#contour = contour[contour[:, 1].argsort()]
'''


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



def find_square():
    """
    f from image return list of contour squares and zoom to board
    :return:
    """
    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))
    contour = big_contour(image)
    countour_origin = big_contour(image)
    angle = protractor(contour)

    contour = np.amin(contour, axis=1)
    contour = contour[contour[:, 1].argsort()]
    TOP = contour[0:2]
    BOTTOM = contour[2:4]
    TOP = TOP[TOP[:, 0].argsort()]
    BOTTOM = BOTTOM[BOTTOM[:, 0].argsort()]

    top_left = TOP[0]
    top_right = TOP[1]
    bottom_left = BOTTOM[0]
    bottom_right = BOTTOM[1]

    # 4 point after

    x, y, w, h = cv2.boundingRect(contour)
    delta = np.array([x, y])
    zoom_contour = contour - delta
    # rotate
    xs = int(w / 2)
    ys = int(h / 2)
    fi = angle * math.pi / 180
    rotate_contour = []
    for point in zoom_contour:
        xp, yp = point[0], point[1]

        xp_r = (xp - xs) * math.cos(fi) - (yp - ys) * math.sin(fi) + xs
        yp_r = (xp - xs) * math.sin(fi) + (yp - ys) * math.cos(fi) + ys

        rotate_contour.append([xp_r, yp_r])

    rotate_contour = np.array(rotate_contour)
    #rotate_contour =
    board = image[y:y+h, x:x+w]
    board = ndimage.rotate(board, angle, cval=255)
    board_rotate = ndimage.rotate(board, 45, cval=255)
    counturs_zoom_reshape = []
    print(zoom_contour[0].shape)

    for i in range(len(zoom_contour)):
        counturs_zoom_reshape.append([zoom_contour[i]])
    counturs_zoom_reshape = np.array(counturs_zoom_reshape)
    cv2.drawContours(board, counturs_zoom_reshape, -1, (0, 255, 0), 3)
    cv2.imshow('img', board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_square2():
    """
    f from image return list of contour squares and zoom to board
    :return:
    """


    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))
    contour = big_contour(image)
    countour_origin = big_contour(image)
    angle = protractor(contour)

    contour = np.amin(contour, axis=1)
    contour = contour[contour[:, 1].argsort()]
    TOP = contour[0:2]
    BOTTOM = contour[2:4]
    TOP = TOP[TOP[:, 0].argsort()]
    BOTTOM = BOTTOM[BOTTOM[:, 0].argsort()]

    top_left = TOP[0]
    top_right = TOP[1]
    bottom_left = BOTTOM[0]
    bottom_right = BOTTOM[1]

    # rotate img and contour
    #angle = 45
    rotate_image = ndimage.rotate(image, angle, cval=0)

    # rotate contour
    rotate_contour = []
    #xs = int(image.shape[1] / 2)
    #ys = int(image.shape[0] / 2)
    xs = int(792 / 2)
    ys = int(792 / 2)
    delta_x = int((rotate_image.shape[1] - image.shape[1]) / 2)
    delta_y = int((rotate_image.shape[0] - image.shape[0]) / 2)

    fi = angle * math.pi / 180
    fi = -1 * fi
    for point in contour:
        xp, yp = point[0] + delta_x, point[1] + delta_y
        xp_r = int((xp - xs) * math.cos(fi) - (yp - ys) * math.sin(fi) + xs)
        yp_r = int((xp - xs) * math.sin(fi) + (yp - ys) * math.cos(fi) + ys)

        rotate_contour.append([[xp_r, yp_r]])
    rotate_contour = np.array(rotate_contour)

    cv2.drawContours(image, countour_origin, -1, (0, 255, 0), 3)
    cv2.imshow('img', image)
    cv2.waitKey(0)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.drawContours(rotate_image, rotate_contour, -1, (0, 255, 0), 3)
    cv2.imshow('img', rotate_image)
    cv2.waitKey(0)




    x, y, w, h = cv2.boundingRect(contour)
    delta = np.array([x, y])
    zoom_contour = contour - delta
    # rotate
    #xs = int(w / 2)
    #ys = int(h / 2)
    xs = int(792 / 2)
    ys = int(792 / 2)

    fi = angle * math.pi / 180
    rotate_contour = []
    for point in zoom_contour:
        xp, yp = point[0], point[1]

        xp_r = (xp - xs) * math.cos(fi) - (yp - ys) * math.sin(fi) + xs
        yp_r = (xp - xs) * math.sin(fi) + (yp - ys) * math.cos(fi) + ys

        rotate_contour.append([xp_r, yp_r])

    rotate_contour = np.array(rotate_contour)
    #rotate_contour =
    board = image[y:y+h, x:x+w]
    board = ndimage.rotate(board, angle, cval=255)
    board_rotate = ndimage.rotate(board, 45, cval=255)
    counturs_zoom_reshape = []
    print(zoom_contour[0].shape)

    for i in range(len(zoom_contour)):
        counturs_zoom_reshape.append([zoom_contour[i]])
    counturs_zoom_reshape = np.array(counturs_zoom_reshape)
    cv2.drawContours(board, counturs_zoom_reshape, -1, (0, 255, 0), 3)
    cv2.imshow('img', board)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#find_square2()

def divide_contour(contour):
    """
    Function divide big contour whole sudoku board into 81 small squares
    :param contour:
    :return:
    """
    top = contour[0:2]
    bottom = contour[2:4]
    top = top[top[:, 0].argsort()]
    bottom = bottom[bottom[:, 0].argsort()]

    top_left = top[0]
    top_right = top[1]
    bottom_left = bottom[0]
    bottom_right = bottom[1]

    top_partition = [top_left]
    left_partition = [top_left]
    right_partition = [top_right]
    bottom_partition = [bottom_left]

    for part in range(1, 10):
        top_partition.append(top_left + (top_right - top_left) * part / 9)
        left_partition.append(top_left + (bottom_left - top_left) * part / 9)
        right_partition.append(top_right + (bottom_right - top_right) * part / 9)
        bottom_partition.append(bottom_left + (bottom_right - bottom_left) * part / 9)

    point_matrix = np.zeros(shape=(10, 10, 2))
    for row in range(1, 9):
        for col in range(1, 9):
            alfa = (bottom_partition[col][1] - top_partition[col][1]) / (
                    bottom_partition[col][0] - top_partition[col][0])
            beta = top_partition[col][1] - (bottom_partition[col][1] - top_partition[col][1]) / (
                    bottom_partition[col][0] - top_partition[col][0]) * top_partition[col][0]
            gamma = (right_partition[row][1] - left_partition[row][1]) / (
                    right_partition[row][0] - left_partition[row][0])
            delta = left_partition[row][1] - (right_partition[row][1] - left_partition[row][1]) / (
                    right_partition[row][0] - left_partition[row][0]) * left_partition[row][0]

            x_coor = (delta - beta) / (alfa - gamma)
            y_coor = alfa * (delta - beta) / (alfa - gamma) + beta

            corner = np.array([x_coor, y_coor])
            point_matrix[row, col] = corner

    point_matrix[0][::] = np.array(top_partition)
    point_matrix[9][::] = np.array(bottom_partition)

    for row in range(10):
        point_matrix[row][0] = np.array(left_partition)[row]
        point_matrix[row][9] = np.array(right_partition)[row]


    contours_sq = []
    for row in range(9):
        for col in range(9):
            contours_sq.append([point_matrix[row][col], point_matrix[row + 1][col], point_matrix[row][col + 1], point_matrix[row + 1][col + 1]])

    contours_sq = np.array(contours_sq)
    counturs_sq_reshape = []
    for i in range(81):
        counturs_sq_reshape.append(contours_sq[i].reshape(4, 1, 2))
    #counturs_sq_reshape = np.array(counturs_sq_reshape)
    return counturs_sq_reshape



def find_square3(image, contour):
    """
    f from image return list of contour squares and zoom to board
    :return:
    """
    # to zas chyba trezbe od komentowac
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #path_read = r"test\catch_board3.jpg"
    #image = cv2.imread(str(path_read))
    #found, contour = big_contour(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = protractor(contour)
    contour = np.amin(contour, axis=1)
    contour = contour[contour[:, 1].argsort()]

    # rotate img
    rotate_image = ndimage.rotate(image, angle, cval=0)

    # rotate contour
    rotate_contour = []
    # center coordinate of image
    xs = int(rotate_image.shape[1] / 2)
    ys = int(rotate_image.shape[0] / 2)

    delta_x = int((rotate_image.shape[1] - image.shape[1]) / 2)
    delta_y = int((rotate_image.shape[0] - image.shape[0]) / 2)

    fi = -angle * math.pi / 180

    for point in contour:
        xp, yp = point[0] + delta_x, point[1] + delta_y
        xp_r = int((xp - xs) * math.cos(fi) - (yp - ys) * math.sin(fi) + xs)
        yp_r = int((xp - xs) * math.sin(fi) + (yp - ys) * math.cos(fi) + ys)
        rotate_contour.append([[xp_r, yp_r]])
    rotate_contour = np.array(rotate_contour)

    '''
    TOP = contour[0:2]
    BOTTOM = contour[2:4]
    TOP = TOP[TOP[:, 0].argsort()]
    BOTTOM = BOTTOM[BOTTOM[:, 0].argsort()]

    top_left = TOP[0]
    top_right = TOP[1]
    bottom_left = BOTTOM[0]
    bottom_right = BOTTOM[1]
    '''

    x, y, w, h = cv2.boundingRect(rotate_contour)
    delta = np.array([x, y])
    zoom_contour = rotate_contour - delta

    zoom_image = rotate_image[y:y+h, x:x+w]
    #counturs_zoom_reshape = []

    #for i in range(len(zoom_contour)):
    #    counturs_zoom_reshape.append([zoom_contour[i]])
    #counturs_zoom_reshape = np.array(counturs_zoom_reshape)

    # crate contours small square
    zoom_contour = np.amin(zoom_contour, axis=1)
    #contour = contour[contour[:, 1].argsort()]

    contours_sq = divide_contour(zoom_contour)
    #contour_test1 = np.array([[[0, 0]], [[0, 100]], [[100, 100]], [[100, 0]]])
    #contour_test2 = np.array([[[150, 150]], [[150, 300]], [[300, 150]], [[300, 300]]])
    #contour_test = [contour_test1, contour_test2]
    #contours_sq = contours_sq.astype(np.int32)
    contours_sq = np.array(contours_sq).astype(np.int32)
    contours_sq = contours_sq.reshape((9, 9, 4, 1, 2))


    #test_con = contours_sq[36]
    #test_con = contours_sq[0].astype(np.int32)
    #print(test_con)
    #my_contour = my_contour.astype(np.int32)

    # To juz jest w funkcji recognize
    #x, y, w, h = cv2.boundingRect(contours_sq[0][3])
    #one_square = zoom_image[y: y + h, x: x + w]

    #cv2.imshow('img', one_square)
    #cv2.drawContours(zoom_image, contours_sq, -1, (0, 255, 0), 1)
    #cv2.imshow('img', zoom_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return zoom_image, contours_sq


def find_square4(image, contour):
    """
    f from image return list of contour squares and zoom to board
    :return:
    """
    # to zas chyba trezbe od komentowac
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #path_read = r"test\catch_board3.jpg"
    #image = cv2.imread(str(path_read))
    #found, contour = big_contour(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = protractor(contour)
    contour = np.amin(contour, axis=1)
    contour = contour[contour[:, 1].argsort()]

    # rotate img
    rotate_image = ndimage.rotate(image, angle, cval=0)

    # rotate contour
    rotate_contour = []
    # center coordinate of image
    xs = int(rotate_image.shape[1] / 2)
    ys = int(rotate_image.shape[0] / 2)

    delta_x = int((rotate_image.shape[1] - image.shape[1]) / 2)
    delta_y = int((rotate_image.shape[0] - image.shape[0]) / 2)

    fi = -angle * math.pi / 180

    for point in contour:
        xp, yp = point[0] + delta_x, point[1] + delta_y
        xp_r = int((xp - xs) * math.cos(fi) - (yp - ys) * math.sin(fi) + xs)
        yp_r = int((xp - xs) * math.sin(fi) + (yp - ys) * math.cos(fi) + ys)
        rotate_contour.append([[xp_r, yp_r]])
    rotate_contour = np.array(rotate_contour)

    x, y, w, h = cv2.boundingRect(rotate_contour)
    delta = np.array([x, y])
    zoom_contour = rotate_contour - delta

    zoom_image = rotate_image[y:y+h, x:x+w]

    # crate contours small square
    zoom_contour = np.amin(zoom_contour, axis=1)

    contours_sq = divide_contour(zoom_contour)
    contours_sq = np.array(contours_sq).astype(np.int32)
    contours_sq = contours_sq.reshape((9, 9, 4, 1, 2))
    #contours_sq = contours_sq.reshape((9, 9))
    splited = True

    return splited, zoom_image, contours_sq

#find_square3()





def test_roate_iamge():
    path_read = r"test\test_rotate.jpg"
    image = cv2.imread(str(path_read))
    angle = 10
    rotate_image = ndimage.rotate(image, angle, cval=0, axes=(0, 1))

    cv2.imshow('img', rotate_image)
    cv2.waitKey(0)
