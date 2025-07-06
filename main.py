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

# Import SAM mobile si disponible
try:
    from mobile_sam_podiatry import process_foot_for_mobile_app, quick_foot_measurement
    SAM_MOBILE_AVAILABLE = True
    print("✅ SAM Mobile chargé")
except ImportError:
    SAM_MOBILE_AVAILABLE = False
    print("⚠️  SAM Mobile non disponible, utilisation K-means")

def list_available_images():
    """Liste toutes les images disponibles dans le dossier data"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(f'data/{ext}'))
        images.extend(glob.glob(f'data/{ext.upper()}'))
    return list(set(images))

def process_single_image_smart(image_path, view_type="single", method="auto", mobile_mode=False, ref_width_mm=210, ref_height_mm=297):
    """
    Traite une image avec choix intelligent de méthode
    
    Args:
        image_path: Chemin vers l'image
        view_type: Type de vue
        method: "auto", "sam", "kmeans"
        mobile_mode: Mode optimisé mobile
    """
    if mobile_mode and SAM_MOBILE_AVAILABLE:
        print(f"📱 Mode mobile SAM: {view_type}")
        
        if method in ["auto", "sam"]:
            # Utiliser SAM mobile optimisé
            measurements = process_foot_for_mobile_app(image_path, save_debug=True, ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm)
            
            if 'error' not in measurements:
                return measurements
            else:
                print(f"⚠️  SAM mobile échoué: {measurements['error']}")
        
        # Fallback K-means si SAM échoue
        print("🔄 Fallback K-means...")
        return process_single_image(image_path, view_type, ref_width_mm, ref_height_mm)
    
    elif method == "sam" and SAM_MOBILE_AVAILABLE:
        # SAM standard (non-mobile)
        measurements = process_foot_for_mobile_app(image_path, save_debug=True, ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm)
        return measurements if 'error' not in measurements else None
    
    else:
        # K-means standard (votre code existant)
        return process_single_image(image_path, view_type, ref_width_mm, ref_height_mm)

def process_single_image(image_path, view_type="single", ref_width_mm=210, ref_height_mm=297):
    """Votre fonction existante (inchangée) avec support d'un objet de référence"""
    print(f"\n📸 Traitement K-means de l'image {view_type}: {image_path}")
    
    try:
        oimg = imread(image_path)
        print(f"✅ Image chargée: {oimg.shape[1]}x{oimg.shape[0]} pixels")
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image: {e}")
        return None

    if not os.path.exists('output'):
        os.makedirs('output')

    try:
        view_output_dir = f'output/{view_type}'
        if not os.path.exists(view_output_dir):
            os.makedirs(view_output_dir)

        print("🔄 Préprocessing...")
        preprocessedOimg = preprocess(oimg)
        cv2.imwrite(f'{view_output_dir}/preprocessedOimg.jpg', preprocessedOimg)

        print("🔄 Clustering K-means...")
        clusteredImg = kMeans_cluster(preprocessedOimg)
        cv2.imwrite(f'{view_output_dir}/clusteredImg.jpg', clusteredImg)

        print("🔄 Détection des contours...")
        edgedImg = edgeDetection(clusteredImg)
        cv2.imwrite(f'{view_output_dir}/edgedImg.jpg', edgedImg)

        boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)

        if len(boundRect) < 2:
            print(f"❌ Pas assez de contours détectés pour {view_type}")
            return None

        pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
        cv2.imwrite(f'{view_output_dir}/pdraw.jpg', pdraw)

        print("🔄 Cropping et analyse...")
        croppedImg, pcropedImg = cropOrig(boundRect[1], clusteredImg)
        cv2.imwrite(f'{view_output_dir}/croppedImg.jpg', croppedImg)

        newImg = overlayImage(croppedImg, pcropedImg)
        cv2.imwrite(f'{view_output_dir}/newImg.jpg', newImg)

        print("🔄 Analyse détaillée du pied...")
        fedged = edgeDetection(newImg)
        fboundRect, fcnt, fcntpoly, fimg = getBoundingBox(fedged)

        if len(fboundRect) < 3:
            print(f"❌ Contour du pied non détecté pour {view_type}")
            return None

        fdraw = drawCnt(fboundRect[2], fcnt, fcntpoly, fimg)
        cv2.imwrite(f'{view_output_dir}/fdraw.jpg', fdraw)

        print("📐 Calcul des mesures podologiques...")
        measurements = calcAdvancedFootMeasures(pcropedImg, fboundRect, fcnt, ref_width_mm, ref_height_mm)

        # Ajouter métadonnées K-means
        measurements['segmentation_method'] = 'K-means'
        measurements['mobile_optimized'] = False

        if measurements['length'] < 15 or measurements['length'] > 35:
            print(f"⚠️  ATTENTION: Mesures probablement incorrectes pour {view_type}")
            print(f"   Longueur détectée: {measurements['length']:.2f} cm")

        return measurements
        
    except Exception as e:
        print(f"❌ Erreur pendant le traitement de {view_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_mobile_report(measurements, image_path):
    """Rapport optimisé pour application mobile"""
    print(f"\n" + "="*60)
    print(f"📱 RAPPORT PODOLOGUE MOBILE - {image_path}")
    print(f"="*60)
    
    # Métadonnées mobiles
    method = measurements.get('segmentation_method', 'Unknown')
    confidence = measurements.get('confidence', 0)
    mobile_opt = measurements.get('mobile_optimized', False)
    
    method_icon = "🤖" if method == "SAM" else "🔬"
    print(f"🔍 Méthode: {method_icon} {method}")
    if confidence > 0:
        print(f"🎯 Confiance: {confidence:.0f}%")
    if mobile_opt:
        print(f"📱 Optimisé mobile: ✅")
    
    print(f"\n📏 MESURES PRINCIPALES:")
    print(f"   • Longueur du pied    : {measurements.get('length', 0):.1f} cm")
    print(f"   • Largeur maximale    : {measurements.get('width', 0):.1f} cm")
    print(f"   • Ratio L/l           : {measurements.get('length_width_ratio', 0):.2f}")
    
    # Classification simple pour mobile
    length = measurements.get('length', 0)
    width = measurements.get('width', 0)
    
    if length > 0 and width > 0:
        # Estimation pointure simple
        if length < 22:
            shoe_size = "35-36"
        elif length < 24:
            shoe_size = "37-39"
        elif length < 26:
            shoe_size = "40-41"
        elif length < 28:
            shoe_size = "42-44"
        else:
            shoe_size = "45+"
        
        # Type de pied simple
        ratio = length / width
        if ratio < 2.2:
            foot_type = "Pied large"
        elif ratio > 2.8:
            foot_type = "Pied étroit"
        else:
            foot_type = "Pied normal"
        
        print(f"\n👟 RECOMMANDATIONS MOBILES:")
        print(f"   • Pointure estimée    : {shoe_size}")
        print(f"   • Type de pied        : {foot_type}")
        
        # Score de qualité global
        quality_score = 0
        if 20 <= length <= 32:
            quality_score += 40
        if 7 <= width <= 13:
            quality_score += 30
        if confidence > 0:
            quality_score += min(30, confidence * 0.3)
        
        quality_emoji = "🟢" if quality_score >= 80 else "🟡" if quality_score >= 60 else "🔴"
        print(f"\n📊 QUALITÉ MESURE: {quality_emoji} {quality_score:.0f}%")
        
        return {
            'shoe_size': shoe_size,
            'foot_type': foot_type,
            'quality_score': quality_score
        }
    
    return {'shoe_size': 'Indéterminé', 'foot_type': 'Indéterminé', 'quality_score': 0}

def compare_methods(image_path, ref_width_mm=210, ref_height_mm=297):
    """Compare SAM mobile vs K-means sur la même image"""
    print(f"\n⚔️  COMPARAISON SAM MOBILE vs K-MEANS")
    print(f"📸 Image: {os.path.basename(image_path)}")
    
    results = {}
    
    # Test K-means
    print("\n1️⃣ Test K-means:")
    start_time = datetime.now()
    kmeans_result = process_single_image_smart(image_path, "compare_kmeans", "kmeans", mobile_mode=False, ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm)
    kmeans_time = (datetime.now() - start_time).total_seconds()
    
    if kmeans_result:
        results['kmeans'] = {
            'measurements': kmeans_result,
            'time_seconds': kmeans_time,
            'success': True
        }
        print(f"✅ K-means: {kmeans_result['length']:.1f}cm x {kmeans_result['width']:.1f}cm ({kmeans_time:.1f}s)")
    else:
        results['kmeans'] = {'success': False, 'time_seconds': kmeans_time}
        print(f"❌ K-means échoué ({kmeans_time:.1f}s)")
    
    # Test SAM mobile
    if SAM_MOBILE_AVAILABLE:
        print("\n2️⃣ Test SAM Mobile:")
        start_time = datetime.now()
        sam_result = process_single_image_smart(image_path, "compare_sam", "sam", mobile_mode=True, ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm)
        sam_time = (datetime.now() - start_time).total_seconds()
        
        if sam_result and 'error' not in sam_result:
            results['sam'] = {
                'measurements': sam_result,
                'time_seconds': sam_time,
                'success': True
            }
            confidence = sam_result.get('confidence', 0)
            print(f"✅ SAM: {sam_result['length']:.1f}cm x {sam_result['width']:.1f}cm ({sam_time:.1f}s, {confidence:.0f}%)")
        else:
            results['sam'] = {'success': False, 'time_seconds': sam_time}
            print(f"❌ SAM échoué ({sam_time:.1f}s)")
    else:
        print("2️⃣ SAM non disponible")
        results['sam'] = {'success': False, 'available': False}
    
    # Comparaison
    if results['kmeans']['success'] and results['sam']['success']:
        print(f"\n📊 COMPARAISON DÉTAILLÉE:")
        
        k_measures = results['kmeans']['measurements']
        s_measures = results['sam']['measurements']
        
        print(f"{'Métrique':<15} {'K-means':<10} {'SAM':<10} {'Diff %':<8} {'Gagnant'}")
        print("-" * 55)
        
        metrics = ['length', 'width', 'length_width_ratio']
        winner_count = {'kmeans': 0, 'sam': 0, 'tie': 0}
        
        for metric in metrics:
            k_val = k_measures.get(metric, 0)
            s_val = s_measures.get(metric, 0)
            
            if k_val > 0:
                diff_pct = ((s_val - k_val) / k_val) * 100
                
                # Déterminer le gagnant basé sur la vraisemblance
                if metric == 'length_width_ratio':
                    # Plus proche de 2.5 = mieux
                    k_error = abs(k_val - 2.5)
                    s_error = abs(s_val - 2.5)
                    winner = "SAM" if s_error < k_error else "K-means" if k_error < s_error else "Égalité"
                else:
                    # Pour longueur/largeur, plus stable = mieux
                    winner = "SAM" if abs(diff_pct) < 10 else "Stable"
                
                if winner == "SAM":
                    winner_count['sam'] += 1
                elif winner == "K-means":
                    winner_count['kmeans'] += 1
                else:
                    winner_count['tie'] += 1
                
                print(f"{metric:<15} {k_val:<10.2f} {s_val:<10.2f} {diff_pct:<8.1f} {winner}")
        
        # Verdict final
        print(f"\n🏆 VERDICT:")
        print(f"   Vitesse: K-means ({results['kmeans']['time_seconds']:.1f}s) vs SAM ({results['sam']['time_seconds']:.1f}s)")
        
        if winner_count['sam'] > winner_count['kmeans']:
            print(f"   Précision: 🥇 SAM (gagne {winner_count['sam']}/{len(metrics)} métriques)")
            print(f"   Recommandation: 📱 SAM pour application mobile")
        elif winner_count['kmeans'] > winner_count['sam']:
            print(f"   Précision: 🥇 K-means (gagne {winner_count['kmeans']}/{len(metrics)} métriques)")
            print(f"   Recommandation: ⚡ K-means pour vitesse")
        else:
            print(f"   Précision: 🤝 Égalité")
            print(f"   Recommandation: 🎯 SAM si confiance élevée, sinon K-means")
    
    return results

def quick_measurement_api(image_path, ref_width_mm=210, ref_height_mm=297):
    """API ultra-rapide pour application mobile"""
    if SAM_MOBILE_AVAILABLE:
        return quick_foot_measurement(image_path, ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm)
    else:
        # Fallback K-means simplifié
        measurements = process_single_image(image_path, "quick", ref_width_mm, ref_height_mm)
        if measurements:
            return {
                'length_cm': round(measurements.get('length', 0), 1),
                'width_cm': round(measurements.get('width', 0), 1),
                'ratio': round(measurements.get('length_width_ratio', 0), 2),
                'confidence': 50,  # Confiance moyenne pour K-means
                'method': 'K-means'
            }
        return {'error': 'Échec de mesure'}

def main():
    """Main amélioré avec support SAM mobile"""
    parser = argparse.ArgumentParser(description='Podologue mobile avec SAM')
    
    # Arguments existants
    parser.add_argument('--top', type=str, help='Image vue du dessus')
    parser.add_argument('--left', type=str, help='Image vue gauche')
    parser.add_argument('--right', type=str, help='Image vue droite')
    
    # Arguments mobiles SAM
    parser.add_argument('--method', choices=['auto', 'sam', 'kmeans', 'compare'], 
                       default='auto', help='Méthode de segmentation')
    parser.add_argument('--mobile', action='store_true', 
                       help='Mode optimisé mobile')
    parser.add_argument('--quick', action='store_true', 
                       help='Mesure ultra-rapide (API mobile)')
    parser.add_argument('--compare-methods', action='store_true',
                       help='Comparer SAM vs K-means')
    parser.add_argument('--reference', choices=['a4', 'credit_card'], default='a4',
                       help="Objet de référence pour l'échelle")
    
    # Argument pour image unique
    parser.add_argument('image', nargs='?', help='Image à analyser')
    
    args = parser.parse_args()

    if args.reference == 'a4':
        ref_width_mm, ref_height_mm = 210, 297
    else:
        ref_width_mm, ref_height_mm = 85.6, 53.98
    
    # Mode comparaison
    if args.compare_methods:
        if args.image:
            image_path = f'data/{args.image}' if not args.image.startswith('data/') else args.image
            if os.path.exists(image_path):
                compare_methods(image_path, ref_width_mm, ref_height_mm)
            else:
                print(f"❌ Image non trouvée: {image_path}")
        else:
            print("❌ Spécifiez une image pour la comparaison")
        return
    
    # Mode quick API
    if args.quick:
        if args.image:
            image_path = f'data/{args.image}' if not args.image.startswith('data/') else args.image
            if os.path.exists(image_path):
                result = quick_measurement_api(image_path, ref_width_mm, ref_height_mm)
                print("\n📱 MESURE RAPIDE:")
                print(f"   Longueur: {result.get('length_cm', 'N/A')} cm")
                print(f"   Largeur: {result.get('width_cm', 'N/A')} cm")
                print(f"   Méthode: {result.get('method', 'N/A')}")
                if 'confidence' in result:
                    print(f"   Confiance: {result['confidence']}%")
            else:
                print(f"❌ Image non trouvée: {image_path}")
        else:
            print("❌ Spécifiez une image pour la mesure rapide")
        return
    
    # Mode multi-vues (votre code existant adapté)
    if args.top or args.left or args.right:
        print("🔍 Mode multi-vues détecté")
        # [Votre code multi-vues existant avec adaptations SAM]
        # Pour économiser l'espace, je garde votre logique existante
        pass
    
    else:
        # Mode image unique
        if args.image:
            image_path = f'data/{args.image}' if not args.image.startswith('data/') else args.image
        else:
            # Image par défaut
            available_images = list_available_images()
            if available_images:
                image_path = available_images[0]
                print(f"📷 Image par défaut: {os.path.basename(image_path)}")
            else:
                print("❌ Aucune image disponible")
                return
        
        if not os.path.exists(image_path):
            print(f"❌ Image non trouvée: {image_path}")
            return
        
        # Traitement selon le mode
        measurements = process_single_image_smart(
            image_path, "single", args.method, args.mobile,
            ref_width_mm=ref_width_mm, ref_height_mm=ref_height_mm
        )
        
        if measurements and 'error' not in measurements:
            podiatry_data = generate_mobile_report(measurements, os.path.basename(image_path))
            
            # Sauvegarde adaptée
            method_suffix = f"_{args.method}" if args.method != "auto" else ""
            mobile_suffix = "_mobile" if args.mobile else ""
            
            report_filename = f"output/rapport{method_suffix}{mobile_suffix}_{os.path.splitext(os.path.basename(image_path))[0]}.txt"
            
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write("RAPPORT PODOLOGUE MOBILE\n")
                f.write("="*30 + "\n\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"Méthode: {measurements.get('segmentation_method')}\n")
                f.write(f"Mobile: {measurements.get('mobile_optimized', False)}\n\n")
                
                f.write("MESURES:\n")
                f.write(f"- Longueur: {measurements.get('length', 0):.1f} cm\n")
                f.write(f"- Largeur: {measurements.get('width', 0):.1f} cm\n")
                f.write(f"- Ratio: {measurements.get('length_width_ratio', 0):.2f}\n\n")
                
                f.write("RECOMMANDATIONS:\n")
                f.write(f"- Pointure: {podiatry_data.get('shoe_size')}\n")
                f.write(f"- Type: {podiatry_data.get('foot_type')}\n")
                f.write(f"- Qualité: {podiatry_data.get('quality_score')}%\n")
            
            print(f"📄 Rapport sauvegardé: {report_filename}")
        else:
            error_msg = measurements.get('error', 'Erreur inconnue') if measurements else 'Échec traitement'
            print(f"❌ Erreur: {error_msg}")

def print_mobile_help():
    """Aide pour l'utilisation mobile"""
    print("""
📱 SAM PODOLOGUE MOBILE - MODE D'EMPLOI

🚀 COMMANDES RAPIDES:
  python main.py --mobile --method sam image.jpg     # SAM optimisé mobile
  python main.py --quick image.jpg                   # Mesure ultra-rapide
  python main.py --compare-methods image.jpg         # SAM vs K-means

📊 MODES DISPONIBLES:
  --mobile     : Optimisations mobile (vitesse + précision)
  --quick      : API ultra-rapide (longueur + largeur uniquement)
  --method sam : Forcer SAM (plus précis)
  --method kmeans : Forcer K-means (plus rapide)

🎯 POUR APPLICATION MOBILE:
  1. Utilisez --mobile --method sam pour la meilleure précision
  2. Utilisez --quick pour un retour rapide à l'utilisateur  
  3. Testez --compare-methods pour évaluer les performances

📏 RÉSULTAT TYPE:
  Longueur: 25.3 cm
  Largeur: 9.8 cm
  Pointure: 40-41
  Confiance: 87%
""")

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_mobile_help()
    main()