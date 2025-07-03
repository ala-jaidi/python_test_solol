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
    #plt.title('Clustered Image')
    plt.show()

def cropOrig(bRect, oimg):
    # x (Horizontal), y (Vertical Downwards) are start coordinates
    # img.shape[0] = height of image
    # img.shape[1] = width of image

    x,y,w,h = bRect

    print(x,y,w,h)
    pcropedImg = oimg[y:y+h,x:x+w]

    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    crop1 = pcropedImg[y1+y2:h1-y2,x1+x2:w1-x2]

    #cv2_imshow(crop1)

    ix, iy, iw, ih = x+x2, y+y2, crop1.shape[1], crop1.shape[0]

    croppedImg = oimg[iy:iy+ih,ix:ix+iw]

    return croppedImg, pcropedImg



def overlayImage(croppedImg, pcropedImg):


    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

    y2 = int(h1/10)

    x2 = int(w1/10)

    new_image = np.zeros((pcropedImg.shape[0], pcropedImg.shape[1], 3), np.uint8)
    new_image[:, 0:pcropedImg.shape[1]] = (255, 0, 0) # (B, G, R)

    new_image[ y1+y2:y1+y2+croppedImg.shape[0], x1+x2:x1+x2+croppedImg.shape[1]] = croppedImg

    return new_image



def kMeans_cluster(img):

    # For clustering the image using k-means, we first need to convert it into a 2-dimensional array
    # (H*W, N) N is channel = 3
    image_2D = img.reshape(img.shape[0]*img.shape[1], img.shape[2])

    # tweak the cluster size and see what happens to the Output
    kmeans = KMeans(n_clusters=2, random_state=0).fit(image_2D)
    clustOut = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape back the image from 2D to 3D image
    clustered_3D = clustOut.reshape(img.shape[0], img.shape[1], img.shape[2])

    clusteredImg = np.uint8(clustered_3D*255)

    return clusteredImg


def edgeDetection(clusteredImage):
  #gray = cv2.cvtColor(hsvImage, cv2.COLOR_BGR2GRAY)
  edged1 = cv2.Canny(clusteredImage, 0, 255)
  edged = cv2.dilate(edged1, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
  return edged

def getBoundingBox(img):

    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #print(len(contours))
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
      #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), \
              #(int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    cv2.rectangle(drawing, (int(paperbb[0]), int(paperbb[1])), \
              (int(paperbb[0]+paperbb[2]), int(paperbb[1]+paperbb[3])), color, 2)
    
    return drawing


def calcFeetSize(pcropedImg, fboundRect):
  x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]

  y2 = int(h1/10)

  x2 = int(w1/10)

  fh = y2 + fboundRect[2][3]
  fw = x2 + fboundRect[2][2]
  ph = pcropedImg.shape[0]
  pw = pcropedImg.shape[1]

  opw = 210
  oph = 297

  ofs = 0.0

  if fw>fh:
    ofs = (opw/pw)*fw
  else :
    ofs = (oph/ph)*fh

  return ofs

def calcFootWidth(pcropedImg, fboundRect, fcnt):
    """
    Calcule la largeur du pied en utilisant le contour détecté
    Retourne la largeur maximale du pied en mm
    """
    x1, y1, w1, h1 = 0, 0, pcropedImg.shape[1], pcropedImg.shape[0]
    
    y2 = int(h1/10)
    x2 = int(w1/10)
    
    # Dimensions du papier A4 en mm
    opw = 210  # largeur A4
    oph = 297  # hauteur A4
    
    # Dimensions du papier en pixels
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    
    # Calcul du ratio pixel/mm
    ratio_x = opw / pw  # mm par pixel en largeur
    ratio_y = oph / ph  # mm par pixel en hauteur
    
    # Utiliser le contour du pied (index 2 généralement)
    if len(fcnt) > 2:
        foot_contour = fcnt[2]
        
        # Calculer la bounding box du contour du pied
        foot_rect = cv2.boundingRect(foot_contour)
        foot_width_pixels = foot_rect[2]  # largeur en pixels
        
        # Convertir en mm
        foot_width_mm = foot_width_pixels * ratio_x
        
        return foot_width_mm
    
    # Fallback: utiliser fboundRect si pas de contour
    fw = x2 + fboundRect[2][2] if len(fboundRect) > 2 else 0
    foot_width_mm = fw * ratio_x
    
    return foot_width_mm

def calcAdvancedFootMeasures(pcropedImg, fboundRect, fcnt):
    """
    Calcule des mesures avancées pour les podologues :
    - Longueur du pied
    - Largeur maximale
    - Largeur au niveau des métatarses (avant-pied)
    - Ratio longueur/largeur
    - Surface du pied
    """
    measurements = {}
    
    # Dimensions du papier A4 en mm
    opw = 210  # largeur A4
    oph = 297  # hauteur A4
    
    # Dimensions du papier en pixels
    ph = pcropedImg.shape[0]
    pw = pcropedImg.shape[1]
    
    # Calcul des ratios pixel/mm
    ratio_x = opw / pw
    ratio_y = oph / ph
    
    # Longueur (déjà calculée dans calcFeetSize)
    measurements['length'] = calcFeetSize(pcropedImg, fboundRect) / 10  # en cm
    
    # Largeur
    measurements['width'] = calcFootWidth(pcropedImg, fboundRect, fcnt) / 10  # en cm
    
    if len(fcnt) > 2:
        foot_contour = fcnt[2]
        
        # Surface du pied
        foot_area_pixels = cv2.contourArea(foot_contour)
        foot_area_cm2 = foot_area_pixels * (ratio_x * ratio_y) / 100  # en cm²
        measurements['area'] = foot_area_cm2
        
        # Calcul de la largeur de l'avant-pied (approximation)
        # On prend la largeur dans le tiers avant du pied
        foot_rect = cv2.boundingRect(foot_contour)
        x, y, w, h = foot_rect
        
        # Zone de l'avant-pied (1/3 avant)
        forefoot_y_start = y + int(h * 0.6)  # 60% depuis le talon
        forefoot_y_end = y + h
        
        # Trouver la largeur maximale dans cette zone
        max_forefoot_width = 0
        for check_y in range(forefoot_y_start, forefoot_y_end, 2):
            # Compter les pixels du contour à cette hauteur
            mask = np.zeros(pcropedImg.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [foot_contour], -1, 255, -1)
            
            if check_y < mask.shape[0]:
                row = mask[check_y, :]
                white_pixels = np.where(row == 255)[0]
                if len(white_pixels) > 0:
                    width_at_y = (white_pixels[-1] - white_pixels[0]) * ratio_x / 10  # en cm
                    max_forefoot_width = max(max_forefoot_width, width_at_y)
        
        measurements['forefoot_width'] = max_forefoot_width
        
    else:
        measurements['area'] = 0
        measurements['forefoot_width'] = 0
    
    # Ratio longueur/largeur
    if measurements['width'] > 0:
        measurements['length_width_ratio'] = measurements['length'] / measurements['width']
    else:
        measurements['length_width_ratio'] = 0
    
    return measurements

def generate_podiatry_report(measurements, image_path):
    """
    Génère un rapport podologique avec toutes les mesures
    """
    print(f"\n" + "="*60)
    print(f"📋 RAPPORT PODOLOGIQUE - {image_path}")
    print(f"="*60)
    
    print(f"📏 MESURES PRINCIPALES:")
    print(f"   • Longueur du pied    : {measurements['length']:.2f} cm")
    print(f"   • Largeur maximale    : {measurements['width']:.2f} cm")
    print(f"   • Largeur avant-pied  : {measurements['forefoot_width']:.2f} cm")
    print(f"   • Surface plantaire   : {measurements['area']:.2f} cm²")
    print(f"   • Ratio L/l           : {measurements['length_width_ratio']:.2f}")
    
    # Initialisation des variables
    foot_type = "Non déterminé"
    shoe_size = "Non déterminé"
    shoe_width = "Non déterminé"
    
    # Classification de la largeur
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
    
    # Estimation pointure
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
            
        # Ajuster la largeur selon le ratio
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