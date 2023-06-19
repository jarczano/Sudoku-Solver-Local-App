import find_sudoku_board
import cv2
import split_board


class TestSudokuBoardVision:
    def test_find_sudoku_board(self):
        VS = 0
        #VS = "http://192.168.1.2:4747/video"
        cap = cv2.VideoCapture(VS)
        found = False

        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            try:
                found, board = find_sudoku_board.find_sudoku_board_2(frame)
            except Exception:
                pass

            if found:
                print("found")
                cv2.imshow('frame', board)
                cv2.waitKey(-1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def test_split_board(self, board):
        """
            Function divides the board into 81 squares
        :param board: image of sudoku board
        :return: if board could  split into 81 squares: split = True and contours_square a list of contours.
                if board could not split into 81 squares: split = False and contours_square None
        """
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        HEIGHT, WIDTH = board.shape[0], board.shape[1]
        HEIGHT_SQ = HEIGHT / 9
        WIDTH_SQ = WIDTH / 9
        AREA_SQ = HEIGHT_SQ * WIDTH_SQ

        # image preprocessing
        # img = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        img = board
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        blur = cv2.medianBlur(blur, 5)
        ret, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)

        # line detection
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(img)[0]
        for line in lines:
            x1, y1, x2, y2 = int(line[0][0]), int(line[0][1]), int(line[0][2]), int(line[0][3])
            cv2.line(thresh, (x1, y1), (x2, y2), (255, 0, 0), 4)

        # contours detection
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cv2.approxPolyDP(contour, HEIGHT_SQ * 0.15, True) for contour in contours]
        contours_square = []

        print("draw")

        # selecting such contours that have: 1) suitable area, 2) 4 vertices, 3) convex shape
        for i in range(len(contours)):
            area = round(cv2.contourArea(contours[i], 1))
            if 0.5 * AREA_SQ < area < 1.5 * AREA_SQ:
                if contours[i].shape[0] == 4:
                    if cv2.isContourConvex(contours[i]):
                        cv2.drawContours(board, contours, i, (255, 0, 0), thickness=4)

                        x, y, w, h = cv2.boundingRect(contours[i])
                        cv2.putText(board, '{} a:{}, p:{}'.format(i, area, contours[i].shape[0]), (x, y + 40), FONT,
                                    0.3, (0, 0, 255), 1)
                        contours_square.append(contours[i])
        #split = True
        cv2.imshow('frame', board)
        cv2.waitKey(-1)
        #if len(contours_square) != 81:
        #    split = False
        #    contours_square = None

        #return split, contours_square

    def test_display_split_board(self):
        VS = 0
        # VS = "http://192.168.1.2:4747/video"
        cap = cv2.VideoCapture(VS)
        found = False

        while True:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            try:
                found, board = find_sudoku_board.find_sudoku_board(frame)
            except Exception:
                pass

            if found:
                print("found")
                TestSudokuBoardVision.test_split_board(frame)
                #cv2.imshow('frame', board)
                #cv2.waitKey(-1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

#test = TestSudokuBoardVision
#test.test_find_sudoku_board(test)
#test.test_display_split_board(test)