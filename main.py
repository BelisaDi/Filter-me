import numpy as np
import cv2
from skimage.util.shape import view_as_windows


import numpy as np
import cv2
from skimage.util.shape import view_as_windows


def hist_equalization(image):
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HLS)

    hist_array = np.bincount(np.asarray(
        image[:, :, 1]).flatten(), minlength=256)
    num_pixels = np.sum(hist_array)
    hist_array = hist_array / num_pixels
    chist_array = np.cumsum(hist_array)

    transform_map = np.floor(255 * chist_array).astype(np.uint8)
    img_list = list(np.asarray(image[:, :, 1]).flatten())

    eq_img_list = [transform_map[p] for p in img_list]

    eq_img_array = np.reshape(np.asarray(eq_img_list),
                              np.asarray(image[:, :, 1]).shape)

    image[:, :, 1] = eq_img_array
    image = cv2.cvtColor(src=image, code=cv2.COLOR_HLS2BGR)

    return image


def convolute(image, kernel):
    output_shape_x = (image.shape[0] - kernel.shape[0]) + 1
    output_shape_y = (image.shape[1] - kernel.shape[1]) + 1
    if len(image.shape) == 2:
        windows = view_as_windows(image, kernel.shape)
        wind_mat = windows.reshape(
            output_shape_x*output_shape_y, kernel.shape[0]*kernel.shape[1]).T
        result = np.dot(kernel.flatten(), wind_mat)
        result = result.reshape(
            output_shape_x, output_shape_y)
        return(result)
    elif len(image.shape) == 3:
        list_matrix = []
        for d in range(3):
            windows = view_as_windows(image[:, :, d], kernel.shape)
            wind_mat = windows.reshape(
                output_shape_x*output_shape_y, kernel.shape[0]*kernel.shape[1]).T
            result = np.dot(kernel.flatten(), wind_mat)
            result = result.reshape(
                output_shape_x, output_shape_y)
            list_matrix.append(result)
        return np.dstack((list_matrix[0], list_matrix[1], list_matrix[2]))
    else:
        print("Invalid dimensions")
        return None


outline = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
blur = np.array([
    [0.0625, 0.125, 0.0625],
    [0.125,  0.25,  0.125],
    [0.0625, 0.125, 0.0625]
])

while(True):
    option = input("Por favor seleccione un tipo de efecto. \n 1: Blur \n 2: Sharpen \n 3: Outline \n 4: Hist Equalization \n")

    try:
        opt = int(option)
        if opt >= 1 and opt <= 4:
            break
        else:
            print("Invalid input, try again")

    except:
        print("Invalid input, try again.")
if opt == 1:
    print("Mientras está en la ventana, presione la letra q para salir")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        frame_padded = cv2.copyMakeBorder(
            frame.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        result = convolute(frame_padded, blur)
        result = cv2.convertScaleAbs(result)

        both = np.hstack((frame, result))

        cv2.imshow("OpenCV", both)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif opt == 2:
    print("Mientras está en la ventana, presione la letra q para salir")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        frame_padded = cv2.copyMakeBorder(
            frame.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        result = convolute(frame_padded, sharpen)
        result = cv2.convertScaleAbs(result)

        both = np.hstack((frame, result))

        cv2.imshow("OpenCV", both)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

elif opt == 3:
    print("Mientras está en la ventana, presione la letra q para salir")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()

        frame_padded = cv2.copyMakeBorder(
            frame.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        result = convolute(frame_padded, outline)
        result = cv2.convertScaleAbs(result)

        both = np.hstack((frame, result))

        cv2.imshow("OpenCV", both)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
elif opt == 4:
    print("Mientras está en la ventana, presione la letra q para salir")
    cap = cv2.VideoCapture(0)

    while(True):
        ret, frame = cap.read()
        result = hist_equalization(frame)
        result = cv2.convertScaleAbs(result)

        both = np.hstack((frame, result))

        cv2.imshow("OpenCV", both)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

else: 
    print("Algo pasó mal :(")
