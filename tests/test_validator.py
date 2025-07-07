import numpy as np
import types, sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import cv2_stub as cv2
sys.modules['cv2'] = cv2

sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch'].cuda = types.SimpleNamespace(is_available=lambda: False)

scipy_mod = types.ModuleType('scipy')
spatial_mod = types.ModuleType('spatial')
distance_mod = types.SimpleNamespace(euclidean=lambda a,b: 0.0)
spatial_mod.distance = distance_mod
scipy_mod.spatial = spatial_mod
sys.modules['scipy'] = scipy_mod
sys.modules['scipy.spatial'] = spatial_mod
sys.modules['scipy.spatial.distance'] = distance_mod
sys.modules['sklearn'] = types.ModuleType('sklearn')
sys.modules['sklearn.cluster'] = types.ModuleType('cluster')
sys.modules['sklearn.cluster'].KMeans = object
sys.modules['imutils'] = types.ModuleType('imutils')
sys.modules['skimage'] = types.ModuleType('skimage')
sys.modules['skimage.io'] = types.ModuleType('io')
sys.modules['skimage.io'].imread = lambda x: None
sys.modules['matplotlib'] = types.ModuleType('matplotlib')
sys.modules['matplotlib.pyplot'] = types.ModuleType('pyplot')

from main import FaceImages


def test_faceimages_validation(tmp_path):
    img = np.zeros((120, 120, 3), dtype=np.uint8)
    f = tmp_path / "img.jpg"
    cv2.imwrite(str(f), img)
    data = {k: str(f) for k in ['top','left','right','front','back']}
    obj = FaceImages(**data)
    assert obj.top == str(f)
