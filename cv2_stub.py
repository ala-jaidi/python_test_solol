import numpy as np

def imwrite(path, img):
    """Minimal stub for cv2.imwrite used in tests."""
    try:
        with open(path, 'wb') as f:
            if hasattr(img, 'tobytes'):
                f.write(img.tobytes())
            else:
                f.write(b'')
        return True
    except Exception:
        return False
