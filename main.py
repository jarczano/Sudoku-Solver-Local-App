import cv2
import copy
import numpy as np

from app.recognize_board import recognize_board
from app.sudoku import Sudoku
from app.image_solve_board import image_solve_board
from app.find_sudoku_board import find_sudoku_board
from app.split_board import split_board, find_squares


def sudoku_solver():
    VS = 0
    cap = cv2.VideoCapture(VS)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow('frame', frame)

        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        if frame_height >= 1080 and frame_width >= 1080:
            high_quality = True
        else:
            high_quality = False

        # Try find the sudoku board on frame
        found = False
        try:
            found, board_contour = find_sudoku_board(frame)
            print("Found: ", found)
        except Exception:
            pass

        if found:

            # Divides sudoku boards into 81 individual fields
            if high_quality:
                splited, board_image, contours_sq = find_squares(frame, board_contour)
            else:
                splited, board_image, contours_sq = split_board(frame, board_contour)
            print("Board splited: ", splited)

            if splited:

                cap.release()

                # Recognize the digits entered in the squares
                digital_board = recognize_board(board_image, contours_sq)

                digital_board_origin = copy.deepcopy(digital_board)

                # Create object Sudoku
                sudoku_read = Sudoku(digital_board)

                sudoku_read.create_board_pos()

                sudoku_read.check_correct()
                print('Sudoku correct:', sudoku_read.correct)

                if sudoku_read.correct:

                    # Solve sudoku
                    success, sudoku_solved = Sudoku.solve(sudoku_read)

                    if success:
                        # Display sudoku solution
                        img_solution = image_solve_board(board_image, contours_sq, digital_board_origin, sudoku_solved)

                        # Resize image solution
                        blank = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                        new_h = frame_height
                        new_w = int(frame_height / img_solution.shape[0] * img_solution.shape[1])
                        resize_img_solution = cv2.resize(img_solution, (new_w, new_h))
                        x_from = int((frame_width - new_w) / 2)
                        x_to = int(x_from + new_w)
                        blank[0:frame_height, x_from: x_to] = resize_img_solution

                        cv2.imshow('frame', blank)
                        cv2.waitKey(-1)

                    else:
                        print('Solve sudoku failed')
                        cap = cv2.VideoCapture(VS)

                    cap = cv2.VideoCapture(VS)

                else:
                    cap = cv2.VideoCapture(VS)

                del sudoku_read

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sudoku_solver()

