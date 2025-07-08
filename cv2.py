"""Lightweight cv2 stub used for tests.

This module provides a subset of the functions and constants from OpenCV so that
the rest of the project can be imported without installing the real package.
The implementations are intentionally minimal and merely avoid AttributeErrors.
They do **not** perform real image processing.
"""

import numpy as np

# --- constants --------------------------------------------------------------
INTER_AREA = 3
INTER_LINEAR = 1
RETR_TREE = 1
RETR_EXTERNAL = 0
CHAIN_APPROX_SIMPLE = 2
COLOR_RGB2HSV = 41
COLOR_BGR2GRAY = 6
COLOR_BGR2RGB = 4
COLOR_RGB2BGR = 4
MORPH_ELLIPSE = 2
MORPH_CLOSE = 3
MORPH_OPEN = 4
FONT_HERSHEY_SIMPLEX = 0

# --- minimal functions ------------------------------------------------------

def imread(path):
    """Return a dummy image array."""
    return np.zeros((120, 120, 3), dtype=np.uint8)


def imwrite(path, img):
    """Pretend to write an image to disk."""
    with open(path, "wb") as f:
        f.write(b"0")
    return True


def cvtColor(img, flag):
    return img


def GaussianBlur(img, ksize, sigmaX=0):
    return img


def Canny(img, th1, th2):
    return np.zeros_like(img)


def dilate(img, kernel=None, iterations=1):
    return img


def erode(img, kernel=None, iterations=1):
    return img


def findContours(img, mode, method):
    return [], None


def contourArea(cnt):
    return 0.0


def approxPolyDP(curve, epsilon, closed):
    return curve


def boundingRect(array):
    return (0, 0, 10, 10)


def drawContours(img, contours, contourIdx, color, thickness=1):
    pass


def rectangle(img, pt1, pt2, color, thickness=1):
    pass


def minAreaRect(contour):
    return ((0, 0), (10, 10), 0)


def boxPoints(rect):
    return np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)


def convexHull(points):
    return points


def getPerspectiveTransform(src, dst):
    return np.eye(3, dtype=np.float32)


def perspectiveTransform(points, m):
    return points


def warpPerspective(src, M, dsize, flags=0):
    return src


def getStructuringElement(shape, ksize):
    return np.ones(ksize, dtype=np.uint8)


def morphologyEx(src, op, kernel, iterations=1):
    return src


def arcLength(curve, closed):
    return 0.0


def fillPoly(img, pts, color):
    pass


def addWeighted(src1, alpha, src2, beta, gamma):
    return src1


def polylines(img, pts, isClosed, color, thickness=1):
    pass


def circle(img, center, radius, color, thickness=1):
    pass


def putText(img, text, org, fontFace, fontScale, color, thickness=1):
    pass

