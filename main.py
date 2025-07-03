from sklearn.cluster import KMeans
import random as rng
import cv2
import imutils
import argparse
from imutils import contours
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
from datetime import datetime

from utils import *

def list_available_images():
    """Liste toutes les images disponibles dans le dossier data"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(f'data/{ext}'))
        images.extend(glob.glob(f'data/{ext.upper()}'))
    return images

def main():
    # Gestion des arguments en ligne de commande
    if len(sys.argv) > 1:
        image_name = sys.argv[1]
        if not image_name.startswith('data/'):
            ImgPath = f'data/{image_name}'
        else:
            ImgPath = image_name
    else:
        # Afficher les images disponibles
        available_images = list_available_images()
        print("Images disponibles dans le dossier 'data/':")
        for i, img in enumerate(available_images, 1):
            print(f"  {i}. {os.path.basename(img)}")
        
        if not available_images:
            print("Aucune image trouvée dans le dossier 'data/'")
            return
        
        # Utiliser la première image par défaut
        ImgPath = available_images[0]
        print(f"\nUtilisation de l'image par défaut: {os.path.basename(ImgPath)}")
        print("Pour utiliser une autre image: python main.py nom_de_l_image.jpg")

    # Vérifier si le fichier existe
    if not os.path.exists(ImgPath):
        print(f"Erreur: L'image '{ImgPath}' n'existe pas!")
        print("Images disponibles:")
        for img in list_available_images():
            print(f"  - {os.path.basename(img)}")
        return

    print(f"\n📸 Traitement de l'image: {ImgPath}")
    
    try:
        oimg = imread(ImgPath)
        print(f"✅ Image chargée: {oimg.shape[1]}x{oimg.shape[0]} pixels")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return

    if not os.path.exists('output'):
        os.makedirs('output')

    try:
        print("🔄 Préprocessing...")
        preprocessedOimg = preprocess(oimg)
        cv2.imwrite('output/preprocessedOimg.jpg', preprocessedOimg)

        print("🔄 Clustering K-means...")
        clusteredImg = kMeans_cluster(preprocessedOimg)
        cv2.imwrite('output/clusteredImg.jpg', clusteredImg)

        print("🔄 Détection des contours...")
        edgedImg = edgeDetection(clusteredImg)
        cv2.imwrite('output/edgedImg.jpg', edgedImg)

        boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
        pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
        cv2.imwrite('output/pdraw.jpg', pdraw)

        print("🔄 Cropping et analyse...")
        croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
        cv2.imwrite('output/croppedImg.jpg', croppedImg)
        
        newImg = overlayImage(croppedImg, pcropedImg)
        cv2.imwrite('output/newImg.jpg', newImg)

        print("🔄 Analyse détaillée du pied...")
        fedged = edgeDetection(newImg)
        fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)
        fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
        cv2.imwrite('output/fdraw.jpg', fdraw)

        # ===== NOUVELLES MESURES PODOLOGIQUES =====
        print("📐 Calcul des mesures podologiques...")
        
        # Calcul de toutes les mesures avancées
        measurements = calcAdvancedFootMeasures(pcropedImg, fboundRect, fcnt)
        
        # Génération du rapport podologique complet
        podiatry_data = generate_podiatry_report(measurements, os.path.basename(ImgPath))
        
        # Sauvegarde du rapport dans un fichier
        report_filename = f"output/rapport_podologique_{os.path.splitext(os.path.basename(ImgPath))[0]}.txt"
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write("RAPPORT PODOLOGIQUE DÉTAILLÉ\n")
            f.write("="*50 + "\n\n")
            f.write(f"Image analysée: {ImgPath}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("MESURES PRINCIPALES:\n")
            f.write(f"- Longueur du pied: {measurements['length']:.2f} cm\n")
            f.write(f"- Largeur maximale: {measurements['width']:.2f} cm\n")
            f.write(f"- Largeur avant-pied: {measurements['forefoot_width']:.2f} cm\n")
            f.write(f"- Surface plantaire: {measurements['area']:.2f} cm²\n")
            f.write(f"- Ratio longueur/largeur: {measurements['length_width_ratio']:.2f}\n\n")
            
            f.write("RECOMMANDATIONS:\n")
            f.write(f"- Pointure suggérée: {podiatry_data['shoe_size']}\n")
            f.write(f"- Largeur suggérée: {podiatry_data['shoe_width']}\n")
            f.write(f"- Type de pied: {podiatry_data['foot_type']}\n\n")
            
            f.write("NOTES:\n")
            f.write("Ces mesures sont calculées automatiquement.\n")
            f.write("Pour un diagnostic médical, consultez un podologue.\n")
        
        print(f"📄 Rapport sauvegardé: {report_filename}")
        print(f"📁 Images de traitement sauvées dans: output/")
        
        # Validation des mesures
        if measurements['length'] < 15 or measurements['length'] > 35:
            print(f"\n⚠️  ATTENTION: Mesures probablement incorrectes")
            print(f"   Vérifiez que l'image respecte les critères:")
            print(f"   - Papier A4 blanc complet visible")
            print(f"   - Pied au centre ne dépassant pas")
            print(f"   - Photo prise du dessus")
            print(f"   - Sol différent du blanc")
        
    except Exception as e:
        print(f"❌ Erreur pendant le traitement: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()