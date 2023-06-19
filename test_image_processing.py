import cv2
import numpy as np
import heapq

def test_image_processing():
    path_read = r"test\catch_board3.jpg"
    image = cv2.imread(str(path_read))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = 255 - image

    image = image / 255

    _, image = cv2.threshold(image, 150 / 255, 250 / 255, cv2.THRESH_BINARY)

    image = cv2.blur(image, (2, 2))  # mozew []
    cv2.imshow('img', image)
    cv2.waitKey(0)



def test_find_contour_shit_camera():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    frame = cv2.imread(str(path_read))

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    blur = cv2.GaussianBlur(img, (7, 7), 0)
    blur = cv2.medianBlur(blur, 3)


    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)
    n_largest = 5
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE   , cv2.CHAIN_APPROX_SIMPLE)



    cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)
    cv2.imshow('img', frame)


    cv2.imshow('img2',thresh)
    cv2.waitKey(0)


    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',thresh)
    cv2.waitKey(0)


    thresh = cv2.dilate(thresh, kernel, iterations=2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',thresh)
    cv2.waitKey(0)


def test_find_contour_good_camera():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    path_read = r"test shit camera/h1.jpg"
    #path_read = r"test shit camera/1.png"
    frame = cv2.imread(str(path_read))

    kernel = np.ones((3, 3), np.uint8)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)



    blur = cv2.GaussianBlur(img, (7, 7), 0)
    blur = cv2.medianBlur(blur, 3)


    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7,2)

    cv2.imshow('img',thresh)
    cv2.waitKey(0)


    thresh = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT, kernel)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',thresh)
    cv2.waitKey(0)


    thresh = cv2.dilate(thresh, kernel, iterations=2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img',thresh)
    cv2.waitKey(0)

#test_find_contour_shit_camera()

def find_cont1():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    filtered = cv2.GaussianBlur(thresh, (3, 3), 0)
    cv2.imshow('img', filtered)
    cv2.waitKey(0)
    contours, _ = cv2.findContours(filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#find_cont1()

def find_cont2():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV) # ten chyba najlepij
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    # Erozja i dylatacja
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    cv2.imshow('img', eroded)
    cv2.waitKey(0)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    cv2.imshow('img', dilated)
    cv2.waitKey(0)
    # Znajdowanie konturów
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_cont3():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    filtered = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 10)
    cv2.imshow('img', filtered)
    cv2.waitKey(0)
    # Progowanie
    _, thresh = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    # Znajdowanie konturów
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_cont4():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(image, 50, 150)
    cv2.imshow('img', edges)
    cv2.waitKey(0)
    # Znajdowanie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_cont5():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/1.png"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    # Filtracja Gaussa
    filtered = cv2.GaussianBlur(thresh, (3, 3), 0)
    cv2.imshow('img', filtered)
    cv2.waitKey(0)
    # Detekcja krawędzi Canny
    edges = cv2.Canny(filtered, 50, 150)
    cv2.imshow('img', edges)
    cv2.waitKey(0)
    # Znajdowanie konturów
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#find_cont4()
# 2 niezle


def find_cont6():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/h4.jpg"
    image = cv2.imread(str(path_read))
    img = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = image.shape[0], image.shape[1]
    img = cv2.resize(img, (640, int(height * 640 / width)))
    image = cv2.resize(image, (640, int(height * 640 / width)))
    #image = cv2.GaussianBlur(image, (3, 3), 0)
    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    image = cv2.medianBlur(image, 3)
    cv2.imshow('img', image)
    cv2.waitKey(0)
    approx_epsilon = 0.05 * 640

    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 2)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', thresh)
    cv2.waitKey(0)
    edges = cv2.Canny(image, 50, 150)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', edges)
    cv2.waitKey(0)
    n_largest = 5
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contours = heapq.nlargest(n_largest, contours, key=cv2.contourArea)

    approx_contour = [cv2.approxPolyDP(largest_contours[0], approx_epsilon, True)]
    #cv2.drawContours(img, approx_contour, 0, (0, 255, 0), 1)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.drawContours(img, approx_contour, -1, (0, 255, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#find_cont5()


def aprove_alg():
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)

    path_read = r"test shit camera/h4.jpg"
    frame = cv2.imread(str(path_read))
    img = frame.copy()


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

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.drawContours(frame, largest_contours, -1, (0, 255, 0), 1)
    cv2.imshow('img', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#aprove_alg()

def scale_contour():
    x = np.array(
    [[[506,  59]],
     [[173, 67]],
     [[168, 399]],
     [[508, 403]]])

    print(x.shape)
    print(2.1 * x)
scale_contour()