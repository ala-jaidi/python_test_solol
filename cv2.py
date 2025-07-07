import numpy as np

def imread(path):
    return np.zeros((120, 120, 3), dtype=np.uint8)

def imwrite(path, img):
    with open(path, 'wb') as f:
        f.write(b'0')
    return True
