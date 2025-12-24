from sklearn.cluster import KMeans
import random as rng
import cv2
import imutils
import argparse
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = img/255
    return img

def plotImage(img):
    plt.imshow(img)
    plt.show()

def cropOrig(bRect, oimg):
    x,y,w,h = bRect
    print(x,y,w,h)
    pcropedImg = oimg[y:y+h,x:x+w]
    
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]
    
    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]
    croppedImg = oimg[iy:iy+ih,ix:ix+iw]
    
    return croppedImg, pcropedImg

def overlayImage(croppedImg, pcropedImg):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    
    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)
    new_image[y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg
    
    return new_image

def kMeans_cluster(img):
    # Convert to 2D array for K-means
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]
    
    # Reshape back to 3D
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])
    clusteredImg = np.uint8(clustered_3D*255)
    
    return clusteredImg

def edgeDetection(clusteredImage):
    edged1 = cv2.Canny(clusteredImage, 0, 255)
    edged = cv2.dilate(edged1, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)
    return edged

def getBoundingBox(img):
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    
    return boundRect, contours, contours_poly, img

def drawCnt(bRect, contours, cntPoly, img):
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    paperbb = bRect
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, cntPoly, i, color)
    
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), 
                  (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    
    return drawing

def keep_foot_only(mask, axis='y'):
    """
    Refine mask to keep only the foot part, removing ankle/leg based on silhouette thickness.
    axis='y': Scans rows (vertical). Assumes leg is at the top (low Y). Good for side view.
    axis='x': Scans columns (horizontal). Assumes leg is at left/right.
    """
    # mask: binary uint8 {0,255} or {0,1}
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    
    # Adaptive kernel size for smoothing (2% of dimension, odd number)
    k_size_y = int(h * 0.02) | 1
    k_size_x = int(w * 0.02) | 1

    if axis == 'y':
        # for each y, thickness in x
        thickness = np.zeros(h, dtype=np.int32)
        for y in range(h):
            xs = np.where(m[y, :] > 0)[0]
            if len(xs) > 0:
                thickness[y] = xs.max() - xs.min()

        # smooth a bit
        thickness_s = cv2.GaussianBlur(thickness.reshape(1,-1).astype(np.float32), (k_size_y, 1), 0).ravel()

        tmax = thickness_s.max()
        if tmax <= 0:
            return mask

        y_peak = int(np.argmax(thickness_s))
        
        # Look "above" the peak (smaller y indices) for the "neck"
        # We want to find where thickness drops significantly (leg width < foot length)
        # Heuristic: < 72% of max thickness (very aggressive to ensure ankle removal)
        
        threshold = 0.72 * tmax
        
        # Check region above peak (0 to y_peak)
        pre_peak = thickness_s[:y_peak]
        cut_candidates = np.where(pre_peak < threshold)[0]
        
        if len(cut_candidates) > 0:
            # We want the cut point closest to the peak (the "neck" junction)
            # This is the largest index in cut_candidates
            y_cut = cut_candidates[-1]
            
            print(f"✂️ Cutting mask at Y={y_cut} (Peak at {y_peak}, MaxW={tmax:.1f}, Thresh={threshold:.1f})")
            
            # Remove everything above y_cut (rows 0 to y_cut)
            m[:y_cut, :] = 0
            return (m * 255).astype(np.uint8)
        else:
            print(f"⚠️ No cut point found (MaxW={tmax:.1f}, Threshold={threshold:.1f})")

    elif axis == 'x':
        # for each x, thickness in y
        thickness = np.zeros(w, dtype=np.int32)
        for x in range(w):
            ys = np.where(m[:, x] > 0)[0]
            if len(ys) > 0:
                thickness[x] = ys.max() - ys.min()

        thickness_s = cv2.GaussianBlur(thickness.reshape(1,-1).astype(np.float32), (k_size_x, 1), 0).ravel()

        tmax = thickness_s.max()
        if tmax <= 0:
            return mask

        x_peak = int(np.argmax(thickness_s))
        threshold = 0.65 * tmax
        
        # Try to find cut on RIGHT side (indices > x_peak)
        right_candidates = np.where(thickness_s[x_peak:] < threshold)[0]
        if len(right_candidates) > 0:
            x_cut = x_peak + int(right_candidates[0])
            print(f"✂️ Cutting mask at X={x_cut} (Right side)")
            m[:, x_cut:] = 0
            return (m * 255).astype(np.uint8)
            
        # Try to find cut on LEFT side (indices < x_peak)
        left_candidates = np.where(thickness_s[:x_peak] < threshold)[0]
        if len(left_candidates) > 0:
            x_cut = left_candidates[-1]
            print(f"✂️ Cutting mask at X={x_cut} (Left side)")
            m[:, :x_cut] = 0
            return (m * 255).astype(np.uint8)

    return mask

def calcFeetSize(pcropedImg, fboundRect):
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    
    fh = y2 + fboundRect[2][3]
    fw = x2 + fboundRect[2][2]
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    
    opw = 210  # A4 width in mm
    oph = 297  # A4 height in mm
    
    ofs = 0.0
    if fw > fh:
        ofs = (opw/pw) * fw
    else:
        ofs = (oph/ph) * fh
    
    return ofs

def calcFootWidth(pcropedImg, fboundRect, fcnt):
    """Calculate foot width using detected contour"""
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    y2 = int(h1/10)
    x2 = int(w1/10)
    
    # A4 dimensions in mm
    opw = 210
    oph = 297
    
    # Paper dimensions in pixels
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    
    # Calculate pixel/mm ratio
    ratio_x = opw / pw
    ratio_y = oph / ph
    
    # Use foot contour (usually index 2)
    if len(fcnt) > 2:
        foot_contour = fcnt[2]
        foot_rect = cv2.boundingRect(foot_contour)
        foot_width_pixels = foot_rect[2]
        foot_width_mm = foot_width_pixels * ratio_x
        return foot_width_mm
    
    # Fallback
    fw = x2 + fboundRect[2][2] if len(fboundRect) > 2 else 0
    foot_width_mm = fw * ratio_x
    return foot_width_mm

def calcFootSideLengths(pcropedImg, foot_contour):
    """Calculate left and right side lengths of foot in cm"""
    opw = 210
    oph = 297
    ph, pw = pcropedImg.shape[:2]
    ratio_x = opw / pw
    ratio_y = oph / ph
    
    contour = foot_contour[:, 0, :]
    top_idx = contour[:,1].argmin()
    bottom_idx = contour[:,1].argmax()
    
    if top_idx < bottom_idx:
        left_pts = contour[top_idx:bottom_idx+1]
        right_pts = np.vstack([contour[bottom_idx:], contour[:top_idx+1]])
    else:
        right_pts = contour[top_idx:bottom_idx+1]
        left_pts = np.vstack([contour[bottom_idx:], contour[:top_idx+1]])
    
    def arc_len(pts):
        length = 0.0
        for i in range(len(pts)-1):
            dx = (pts[i+1][0] - pts[i][0]) * ratio_x
            dy = (pts[i+1][1] - pts[i][1]) * ratio_y
            length += (dx**2 + dy**2) ** 0.5
        return length / 10
    
    return arc_len(left_pts), arc_len(right_pts)

def calcAdvancedFootMeasures(pcropedImg, fboundRect, fcnt):
    """Calculate advanced foot measurements for podiatrists"""
    measurements = {}
    
    # A4 dimensions in mm
    opw = 210
    oph = 297
    
    # Paper dimensions in pixels
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    
    # Calculate pixel/mm ratios
    ratio_x = opw / pw
    ratio_y = oph / ph
    
    # Length (from calcFeetSize)
    measurements['length'] = calcFeetSize(pcropedImg, fboundRect) / 10  # in cm
    
    # Width
    measurements['width'] = calcFootWidth(pcropedImg, fboundRect, fcnt) / 10  # in cm
    
    if len(fcnt) > 2:
        foot_contour = fcnt[2]
        
        # Foot area
        foot_area_pixels = cv2.contourArea(foot_contour)
        foot_area_cm2 = foot_area_pixels * (ratio_x * ratio_y) / 100  # in cm²
        measurements['area'] = foot_area_cm2
        
        # Forefoot width calculation
        foot_rect = cv2.boundingRect(foot_contour)
        x, y, w, h = foot_rect
        
        # Forefoot zone (front third)
        forefoot_y_start = y + int(h * 0.6)  # 60% from heel
        forefoot_y_end = y + h
        
        # Find maximum width in this zone
        max_forefoot_width = 0
        for check_y in range(forefoot_y_start, forefoot_y_end, 2):
            mask = np.zeros(pcropedImg.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [foot_contour], -1, 255, -1)
            
            if check_y < mask.shape[0]:
                row = mask[check_y, :]
                white_pixels = np.where(row == 255)[0]
                if len(white_pixels) > 0:
                    width_at_y = (white_pixels[-1] - white_pixels[0]) * ratio_x / 10  # in cm
                    max_forefoot_width = max(max_forefoot_width, width_at_y)
        
        measurements['forefoot_width'] = max_forefoot_width
        
        # Side lengths
        left_length, right_length = calcFootSideLengths(pcropedImg, foot_contour)
        measurements['left_side_length'] = left_length
        measurements['right_side_length'] = right_length
        
    else:
        measurements['area'] = 0
        measurements['forefoot_width'] = 0
        measurements['left_side_length'] = 0
        measurements['right_side_length'] = 0
    
    # Length/width ratio
    if measurements['width'] > 0:
        measurements['length_width_ratio'] = measurements['length'] / measurements['width']
    else:
        measurements['length_width_ratio'] = 0
    
    return measurements

def generate_podiatry_report(measurements, image_path):
    """Generate a complete podiatry report with all measurements"""
    print(f"\n" + "="*60)
    print(f"📋 RAPPORT PODOLOGIQUE - {image_path}")
    print(f"="*60)
    
    print(f"📏 MESURES PRINCIPALES:")
    print(f"   • Longueur du pied    : {measurements['length']:.2f} cm")
    print(f"   • Largeur maximale    : {measurements['width']:.2f} cm")
    print(f"   • Largeur avant-pied  : {measurements['forefoot_width']:.2f} cm")
    print(f"   • Surface plantaire   : {measurements['area']:.2f} cm²")
    print(f"   • Ratio L/l           : {measurements['length_width_ratio']:.2f}")
    print(f"   • Longueur côté gauche : {measurements['left_side_length']:.2f} cm")
    print(f"   • Longueur côté droit  : {measurements['right_side_length']:.2f} cm")
    
    # Initialize variables
    foot_type = "Non déterminé"
    shoe_size = "Non déterminé"
    shoe_width = "Non déterminé"
    
    # Width classification
    if measurements['length'] > 0:
        width_percentage = (measurements['width'] / measurements['length']) * 100
        print(f"\n🦶 CLASSIFICATION:")
        print(f"   • Pourcentage largeur : {width_percentage:.1f}%")
        
        if width_percentage < 35:
            foot_type = "Pied étroit"
        elif width_percentage < 42:
            foot_type = "Pied normal"
        else:
            foot_type = "Pied large"
        print(f"   • Type de pied        : {foot_type}")
    
    # Shoe size estimation
    length = measurements['length']
    if length > 20:
        if length < 22:
            shoe_size = "35-36"
            shoe_width = "D-E"
        elif length < 23:
            shoe_size = "37-38"
            shoe_width = "D-E"
        elif length < 24:
            shoe_size = "38-39"
            shoe_width = "D-E"
        elif length < 25:
            shoe_size = "39-40"
            shoe_width = "D-E"
        elif length < 26:
            shoe_size = "40-41"
            shoe_width = "E-F"
        elif length < 27:
            shoe_size = "42-43"
            shoe_width = "E-F"
        elif length < 28:
            shoe_size = "43-44"
            shoe_width = "F"
        else:
            shoe_size = "45+"
            shoe_width = "F-G"
        
        # Adjust width according to ratio
        if measurements['width'] > 0 and measurements['length'] > 0:
            width_percentage = (measurements['width'] / measurements['length']) * 100
            if width_percentage > 42:
                shoe_width = shoe_width.replace("D", "E").replace("E", "F").replace("F", "G")
            elif width_percentage < 35:
                shoe_width = shoe_width.replace("F", "E").replace("E", "D").replace("G", "F")
    else:
        shoe_size = "Mesure incorrecte"
        shoe_width = "N/A"
    
    print(f"\n👟 RECOMMANDATIONS CHAUSSURES:")
    print(f"   • Pointure suggérée   : {shoe_size}")
    print(f"   • Largeur suggérée    : {shoe_width}")
    
    print(f"\n💡 NOTES PODOLOGIQUES:")
    if measurements['forefoot_width'] > 0 and measurements['width'] > 0 and measurements['forefoot_width'] > measurements['width'] * 0.8:
        print(f"   • Avant-pied développé - Considérer chaussures à bout large")
    if measurements['length_width_ratio'] > 2.8:
        print(f"   • Pied élancé - Attention aux frottements latéraux")
    if measurements['length_width_ratio'] < 2.2:
        print(f"   • Pied trapu - Surveiller les appuis")
    
    print(f"="*60)
    
    return {
        'shoe_size': shoe_size,
        'shoe_width': shoe_width,
        'foot_type': foot_type
    }


def detect_credit_card_reference(image):
    """
    Détecte la carte de crédit dans l'image via OpenCV (contours).
    Retourne le rectangle détecté et le ratio pixel/mm.
    """
    # Prétraitement
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)

    # Recherche de contours
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    best_score = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000:  # Trop petit
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # Rectangle
            rect = cv2.boundingRect(approx)
            w, h = rect[2], rect[3]
            ratio = max(w, h) / min(w, h)
            ratio_diff = abs(ratio - 1.586)
            if ratio_diff < 0.2 and area > best_score:
                best_score = area
                best_rect = rect

    if best_rect is not None:
        w, h = best_rect[2], best_rect[3]
        # Choisir l'orientation correcte
        width_mm = 85.60
        height_mm = 53.98
        if w > h:
            ratio_px_mm = w / width_mm
        else:
            ratio_px_mm = h / width_mm
        return {
            'rect': best_rect,
            'ratio_px_mm': ratio_px_mm,
            'success': True
        }
    else:
        return {
            'success': False,
            'error': "Carte de crédit non détectée"
        }

def detectAndCalibrateReference(pcropedImg, contours):
    """
    Détecte une carte de crédit comme gabarit de référence
    et calcule le ratio pixels/mm avec debug visuel.
    """
    print("🔍 Recherche de carte de crédit...")
    best_contour = None
    best_score = 0

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        rect = cv2.minAreaRect(c)
        (w, h) = rect[1]
        if w == 0 or h == 0:
            continue
        ratio = max(w, h) / min(w, h)
        ratio_diff = abs(ratio - 1.586)
        if ratio_diff < 0.3:
            score = 100 - ratio_diff * 100
            if score > best_score:
                best_score = score
                best_contour = c

    if best_contour is not None and best_score >= 70:
        box = cv2.minAreaRect(best_contour)
        box_points = cv2.boxPoints(box)
        box_points = np.int0(box_points)
        width_px = max(box[1])
        height_px = min(box[1])
        ratio_x = 85.60 / width_px
        ratio_y = 53.98 / height_px

        debug = pcropedImg.copy()
        cv2.drawContours(debug, [box_points], 0, (0, 0, 255), 2)
        cv2.imwrite("output/DEBUG_reference_detected.jpg", debug)

        print(f"✅ Carte détectée : {width_px:.1f}px → 85.6mm")
        return {
            'ratio_x': ratio_x,
            'ratio_y': ratio_y,
            'confidence_score': best_score,
            'reference_type': 'credit_card_detected'
        }
    else:
        print("⚠️ Aucune carte fiable détectée, fallback par défaut.")
        ph, pw = pcropedImg.shape[:2]
        return {
            'ratio_x': 85.60 / (pw * 0.3),
            'ratio_y': 53.98 / (ph * 0.2),
            'confidence_score': 30,
            'reference_type': 'credit_card_default'
        }


def calcRobustFootMeasures(pcropedImg, fboundRect, fcnt, force_credit_card=True):
    """
    Mesure robuste avec calibration par carte de crédit détectée.
    """
    print("🏆 Mode calibrage : Carte de crédit réelle")
    bound = fboundRect[2]
    contour = fcnt[2]
    ratio = detectAndCalibrateReference(pcropedImg, fcnt)
    ratio_x, ratio_y = ratio['ratio_x'], ratio['ratio_y']

    length = bound[3] * ratio_y / 10
    width = bound[2] * ratio_x / 10
    area = cv2.contourArea(contour) * ratio_x * ratio_y / 100

    return {
        'length': length,
        'width': width,
        'area': area,
        'length_width_ratio': length / width if width > 0 else 0,
        'calibration_info': ratio
    }


def analyzeHeelShapeRobust(foot_contour, ratio_x, ratio_y):
    """Version robuste de l'analyse du talon"""
    contour_points = foot_contour[:, 0, :]
    
    # Zone du talon (30% arrière)
    min_y = contour_points[:, 1].min()
    max_y = contour_points[:, 1].max()
    heel_threshold = max_y - (max_y - min_y) * 0.3
    
    heel_points = contour_points[contour_points[:, 1] >= heel_threshold]
    
    if len(heel_points) < 5:
        return {'heel_width_cm': 0, 'heel_shape': 'Non analysable'}
    
    # Largeur du talon avec calibrage précis
    heel_width_px = heel_points[:, 0].max() - heel_points[:, 0].min()
    heel_width_cm = (heel_width_px * ratio_x) / 10
    
    # Analyse de forme améliorée
    try:
        ellipse = cv2.fitEllipse(heel_points.reshape(-1, 1, 2))
        (center_x, center_y), (axis1, axis2), angle = ellipse
        ellipse_ratio = max(axis1, axis2) / min(axis1, axis2)
        
        if ellipse_ratio > 2.5:
            heel_shape = "Talon étroit"
        elif ellipse_ratio > 1.8:
            heel_shape = "Talon normal"
        else:
            heel_shape = "Talon large"
            
        roundness = max(0, 1 - (ellipse_ratio - 1) / 2)
        
    except:
        heel_shape = "Forme standard"
        roundness = 0.5
    
    return {
        'heel_width_cm': round(heel_width_cm, 1),
        'heel_shape': heel_shape,
        'heel_roundness': round(roundness, 2)
    }

def estimateInstepHeightRobust(foot_contour, ratio_x, ratio_y):
    """Estimation robuste du cou-de-pied"""
    contour_points = foot_contour[:, 0, :]
    
    # Analyse de courbure dans zone cou-de-pied
    min_y = contour_points[:, 1].min()
    max_y = contour_points[:, 1].max()
    foot_length = max_y - min_y
    
    instep_start = min_y + foot_length * 0.4
    instep_end = min_y + foot_length * 0.6
    
    instep_points = contour_points[
        (contour_points[:, 1] >= instep_start) & 
        (contour_points[:, 1] <= instep_end)
    ]
    
    if len(instep_points) == 0:
        return {
            'instep_height_estimate_cm': 4.5,
            'instep_category': 'Estimation standard'
        }
    
    # Variation de largeur = indicateur de hauteur
    widths = []
    for y in range(int(instep_start), int(instep_end), 3):
        level_points = instep_points[np.abs(instep_points[:, 1] - y) <= 1]
        if len(level_points) >= 2:
            width = level_points[:, 0].max() - level_points[:, 0].min()
            widths.append(width * ratio_x / 10)  # en cm
    
    if len(widths) > 0:
        width_variation = np.std(widths) / np.mean(widths)
        height_estimate = 4.0 + (width_variation * 3)  # 4-7 cm selon variation
        
        if height_estimate > 6:
            category = "Cou-de-pied haut probable"
        elif height_estimate > 5:
            category = "Cou-de-pied normal-haut"
        elif height_estimate > 3.5:
            category = "Cou-de-pied normal"
        else:
            category = "Cou-de-pied bas probable"
    else:
        height_estimate = 4.5
        category = "Estimation standard"
    
    return {
        'instep_height_estimate_cm': round(height_estimate, 1),
        'instep_category': category
    }

def analyzeArchSupportRobust(foot_contour, ratio_x, ratio_y):
    """Analyse robuste de la voûte plantaire"""
    contour_points = foot_contour[:, 0, :]
    
    # Zone voûte (milieu du pied)
    min_y = contour_points[:, 1].min()
    max_y = contour_points[:, 1].max()
    foot_length = max_y - min_y
    
    arch_start = min_y + foot_length * 0.3
    arch_end = min_y + foot_length * 0.7
    
    arch_points = contour_points[
        (contour_points[:, 1] >= arch_start) & 
        (contour_points[:, 1] <= arch_end)
    ]
    
    if len(arch_points) == 0:
        return {'arch_type': 'Non déterminé'}
    
    # Largeur de la voûte
    arch_width = arch_points[:, 0].max() - arch_points[:, 0].min()
    total_width = contour_points[:, 0].max() - contour_points[:, 0].min()
    
    arch_ratio = arch_width / total_width if total_width > 0 else 0
    
    # Classification précise
    if arch_ratio > 0.85:
        arch_type = "Pied plat - Support voûte fort requis"
    elif arch_ratio > 0.75:
        arch_type = "Voûte basse - Support modéré requis"
    elif arch_ratio > 0.60:
        arch_type = "Voûte normale - Support léger"
    elif arch_ratio > 0.45:
        arch_type = "Voûte haute - Amortissement avant-pied"
    else:
        arch_type = "Pied creux - Amortissement généralisé"
    
    return {
        'arch_type': arch_type,
        'arch_ratio': round(arch_ratio, 2)
    }