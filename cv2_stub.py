import os

def imwrite(path, img):
    with open(path, 'wb') as f:
        if hasattr(img, 'tobytes'):
            f.write(img.tobytes())
        else:
            f.write(b'')
    return True
