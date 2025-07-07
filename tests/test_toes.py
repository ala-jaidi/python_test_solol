import numpy as np
import pytest
import types, sys

sys.modules['torch'] = types.ModuleType('torch')
sys.modules['torch'].cuda = types.SimpleNamespace(is_available=lambda: False)

scipy_mod = types.ModuleType('scipy')
spatial_mod = types.ModuleType('spatial')
def euclid(a,b):
    a=np.array(a);b=np.array(b)
    return float(np.sqrt(((a-b)**2).sum()))
distance_mod = types.SimpleNamespace(euclidean=euclid)
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

from mobile_sam_podiatry import MobileSAMPodiatryPipeline


def test_toe_distance():
    pipe = MobileSAMPodiatryPipeline.__new__(MobileSAMPodiatryPipeline)
    contour = np.array([[[0,0]], [[5,20]], [[10,0]]], dtype=np.int32)
    heel = np.array([5,20])
    toe = np.array([5,0])
    res = pipe._analyze_toes(contour, heel, toe, ratio_px_mm=1.0)
    assert res['bigtoe_to_heel_cm'] == pytest.approx(2.06, rel=1e-2)
    assert res['littletoe_to_heel_cm'] == pytest.approx(2.06, rel=1e-2)
