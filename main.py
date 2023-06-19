import find_sudoku_board
import utils
import cv2
import copy
import recognize_board
import split_board
import sudoku
import image_solve_board
from test_divide_board import big_contour2, find_square3, find_square4
from flask import Flask, Response


def sudoku_solver():
    VS = 0
    video = r'videoCapture/VID_20230523_142658.mp4'
    #VS = "http://192.168.1.2:4747/video"
    cap = cv2.VideoCapture(video)
    #cap = cv2.VideoCapture(VS, cv2.CAP_DSHOW)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    found = False # if not should exist another found = False on the end of while ?

    while True:

        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # Try find the sudoku board on frame
        try:
            found, board = find_sudoku_board.find_sudoku_board(frame)
        except Exception:
            pass

        if found:
            print("found")
            # Divide the flat sudoku into 81 squares.
            split, contours_square = split_board.split_board(board)
            if split:
                print("split")
                cap.release()

                # Sort 81 squares from top left to bottom right
                sorted_split_board = utils.sorted_squares(contours_square)

                # Recognize the digits entered in the squares
                digital_board = recognize_board.recognize_board(board, sorted_split_board)

                digital_board_origin = copy.deepcopy(digital_board)

                # Create object Sudoku
                sudoku_read = sudoku.Sudoku(digital_board)
                sudoku_read.create_board_pos()
                sudoku_read.check_correct()
                print("correct",sudoku_read.correct)

                if sudoku_read.correct:
                    # Solve sudoku
                    sudoku_solved = sudoku.Sudoku.solve(sudoku_read)

                    # Display sudoku solution
                    img_solution = image_solve_board.image_solve_board(board, sorted_split_board, digital_board_origin, sudoku_solved)
                    cv2.imshow('frame', img_solution)
                    cv2.waitKey(-1)

                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()

                else:
                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()
                del sudoku_read

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def sudoku_solver2():
    #video = r'videoCapture/VID_20230523_141923.mp4'
    video = r'videoCapture/VID_20230523_142658.mp4'
    VS = 0
    cap = cv2.VideoCapture(0)

    found = False # if not should exist another found = False on the end of while ?

    while True:

        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        # Try find the sudoku board on frame
        try:
            found, board_contour = big_contour2(frame)
        except Exception:
            pass

        if found:
            print("found")
            # Divide the flat sudoku into 81 squares.

            board_image, contours_sq = find_square3(frame, board_contour)

            cap.release()

            # Sort 81 squares from top left to bottom right
            #sorted_split_board = utils.sorted_squares(contours_square)

            # Recognize the digits entered in the squares
            digital_board = recognize_board.recognize_board(board_image, contours_sq)

            digital_board_origin = copy.deepcopy(digital_board)

            # Create object Sudoku
            sudoku_read = sudoku.Sudoku(digital_board)

            sudoku_read.create_board_pos()

            sudoku_read.check_correct()

            if sudoku_read.correct:

                # Solve sudoku
                sudoku_solved = sudoku.Sudoku.solve(sudoku_read)

                # Display sudoku solution
                img_solution = image_solve_board.image_solve_board(board_image, contours_sq, digital_board_origin, sudoku_solved)
                cv2.imshow('frame', img_solution)
                cv2.waitKey(-1)

                cap = cv2.VideoCapture(VS)
                #ret, frame = cap.read()

            else:
                cap = cv2.VideoCapture(VS)
                #ret, frame = cap.read()
            del sudoku_read

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def sudoku_solver3():
    #video = r'videoCapture/VID_20230523_141923.mp4'
    video = r'videoCapture/VID_20230523_142658.mp4'
    VS = 0
    cap = cv2.VideoCapture(video)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:

        ret, frame = cap.read()
        cv2.imshow('frame', frame)

        if frame.shape[0] >= 1080 and frame.shape[1] >= 1920:
            high_quality = True
        else:
            high_quality = False

        # Try find the sudoku board on frame
        found = False
        try:
            found, board_contour = big_contour2(frame)
        except Exception:
            pass

        if found:

            print("found")

            if high_quality:
                splited, board_image, contours_sq = split_board.split_board2(frame, board_contour)
            else:
                splited, board_image, contours_sq = find_square4(frame, board_contour)

            if splited:
                cap.release()

                # Recognize the digits entered in the squares
                digital_board = recognize_board.recognize_board2(board_image, contours_sq)

                digital_board_origin = copy.deepcopy(digital_board)

                # Create object Sudoku
                sudoku_read = sudoku.Sudoku(digital_board)

                sudoku_read.create_board_pos()

                sudoku_read.check_correct()
                print('correct:', sudoku_read.correct)
                if sudoku_read.correct:

                    # Solve sudoku
                    sudoku_solved = sudoku.Sudoku.solve(sudoku_read)

                    # Display sudoku solution
                    img_solution = image_solve_board.image_solve_board(board_image, contours_sq, digital_board_origin, sudoku_solved)
                    cv2.imshow('frame', img_solution)
                    cv2.waitKey(-1)

                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()

                else:
                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()
                del sudoku_read

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def sudoku_solver4():
    # Version to flask
    #video = r'videoCapture/VID_20230523_141923.mp4'
    video = r'videoCapture/VID_20230523_142658.mp4'
    VS = 0
    cap = cv2.VideoCapture(video)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    while True:

        ret, frame = cap.read()
        #cv2.imshow('frame', frame)

        if frame.shape[0] >= 1080 and frame.shape[1] >= 1920:
            high_quality = True
        else:
            high_quality = False

        # Try find the sudoku board on frame
        found = False
        try:
            found, board_contour = big_contour2(frame)
        except Exception:
            pass

        if found:

            print("found")

            if high_quality:
                splited, board_image, contours_sq = split_board.split_board2(frame, board_contour)
            else:
                splited, board_image, contours_sq = find_square4(frame, board_contour)

            if splited:
                cap.release()

                # Recognize the digits entered in the squares
                digital_board = recognize_board.recognize_board2(board_image, contours_sq)

                digital_board_origin = copy.deepcopy(digital_board)

                # Create object Sudoku
                sudoku_read = sudoku.Sudoku(digital_board)

                sudoku_read.create_board_pos()

                sudoku_read.check_correct()
                print('correct:', sudoku_read.correct)
                if sudoku_read.correct:

                    # Solve sudoku
                    sudoku_solved = sudoku.Sudoku.solve(sudoku_read)

                    # Display sudoku solution
                    img_solution = image_solve_board.image_solve_board(board_image, contours_sq, digital_board_origin, sudoku_solved)
                    cv2.imshow('frame', img_solution)
                    cv2.waitKey(-1)

                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()

                else:
                    cap = cv2.VideoCapture(VS)
                    #ret, frame = cap.read()
                del sudoku_read


        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    cap.release()
    cv2.destroyAllWindows()


app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    return Response(sudoku_solver4(), mimetype='multipart/x-mixed-replace; boundary=frame')




if __name__ == '__main__':
    #sudoku_solver3()
    app.run(debug=True)

