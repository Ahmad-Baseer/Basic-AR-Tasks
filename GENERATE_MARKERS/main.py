import cv2 as cv
from cv2 import aruco
import numpy as np

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = cv.aruco.ArucoDetector(dictionary, parameters)

MARKER_SIZE = 400  # pixels
SQUARE_LENGTH = int(MARKER_SIZE / 5)
MARKER_LENGTH = int(SQUARE_LENGTH * 3 / 4)

for id in range(20):
    diamond_id = np.array([id, id + 1, id + 2, id + 3])
    marker_image = aruco.drawCharucoDiamond(dictionary, diamond_id, SQUARE_LENGTH, MARKER_LENGTH)
    # cv.imshow("img", marker_image)
    cv.imwrite(f"markers/markder_{id}.png", marker_image)
    # cv.waitKey(0)
    # break
