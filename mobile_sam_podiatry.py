# mobile_sam_podiatry.py - SAM optimisé pour application mobile podologue

import cv2
import numpy as np
import os
import torch
from datetime import datetime

# Import des fonctions existantes
from utils import *

# SAM imports avec gestion d'erreur
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

class MobileSAMFootSegmenter:
    def __init__(self, model_type="vit_b"):
        """
        SAM optimisé pour application mobile podologue
        - Segmentation précise pied (cheville -> gros orteil)
        - Gestion multi-angles
        - Optimisé pour vitesse mobile
        """
        self.initialized = False
        
        if not SAM_AVAILABLE:
            print("⚠️  SAM non disponible, utilisation K-means fallback")
            return
        
        # Configuration mobile (plus rapide)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        
        # Charger SAM
        checkpoint_path = self._get_or_download_checkpoint()
        if checkpoint_path:
            try:
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                self.sam.to(device=self.device)
                
                # Configurateur SAM optimisé pour pieds
                self.predictor = SamPredictor(self.sam)
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=16,        # Réduit pour vitesse mobile
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.90,
                    crop_n_layers=0,           # Pas de crop pour vitesse
                    min_mask_region_area=1500, # Taille minimum pied
                )
                
                self.initialized = True
                print(f"✅ SAM Mobile initialisé ({self.device})")
                
            except Exception as e:
                print(f"❌ Erreur SAM: {e}")
                self.initialized = False
    
    def _get_or_download_checkpoint(self):
        """Gère le téléchargement automatique du checkpoint"""
        checkpoint_dir = "sam_mobile"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/sam_{self.model_type}_mobile.pth"
        
        if not os.path.exists(checkpoint_path):
            print(f"📥 Téléchargement SAM {self.model_type} pour mobile...")
            
            urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
            
            try:
                import requests
                response = requests.get(urls[self.model_type], stream=True)
                response.raise_for_status()
                
                with open(checkpoint_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"✅ SAM téléchargé: {checkpoint_path}")
                return checkpoint_path
            except Exception as e:
                print(f"❌ Erreur téléchargement: {e}")
                return None
        
        return checkpoint_path
    
    def segment_foot_smart(self, image, reference_detection=True):
        """
        Segmentation intelligente du pied pour application podologue
        
        Args:
            image: Image à segmenter
            reference_detection: Détecter automatiquement la référence (A4/carte)
        
        Returns:
            dict: {
                'foot_mask': masque du pied,
                'reference_mask': masque de référence,
                'confidence': score de confiance,
                'method_used': 'SAM' ou 'K-means'
            }
        """
        result = {
            'foot_mask': None,
            'reference_mask': None,
            'confidence': 0,
            'method_used': 'K-means'  # Fallback par défaut
        }
        
        # Essayer SAM d'abord
        if self.initialized:
            try:
                print("🤖 Segmentation SAM optimisée...")
                
                # Générer tous les masques
                masks = self.mask_generator.generate(image)
                
                if masks:
                    # Analyser et séparer pied vs référence
                    foot_mask, ref_mask, confidence = self._analyze_masks_for_podiatry(masks, image)
                    
                    if foot_mask is not None:
                        result.update({
                            'foot_mask': foot_mask,
                            'reference_mask': ref_mask,
                            'confidence': confidence,
                            'method_used': 'SAM'
                        })
                        print(f"✅ SAM réussi (confiance: {confidence:.1f}%)")
                        return result
                
            except Exception as e:
                print(f"⚠️  SAM échoué: {e}")
        
        # Fallback K-means
        print("🔄 Fallback K-means...")
        foot_mask = self._kmeans_fallback(image)
        if foot_mask is not None:
            result.update({
                'foot_mask': foot_mask,
                'confidence': 50,  # Confiance moyenne pour K-means
                'method_used': 'K-means'
            })
        
        return result
    
    def _analyze_masks_for_podiatry(self, masks, image):
        """
        Analyse les masques SAM pour identifier pied vs référence (A4/carte)
        Spécialement conçu pour application podologue
        """
        h, w = image.shape[:2]
        image_area = h * w
        
        foot_candidates = []
        reference_candidates = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']
            stability = mask_data['stability_score']
            
            # Calculs géométriques
            area_ratio = area / image_area
            bbox_w, bbox_h = bbox[2], bbox[3]
            aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if min(bbox_w, bbox_h) > 0 else 0
            
            # Classification pied vs référence
            if self._is_foot_candidate(area_ratio, aspect_ratio, bbox, h, w, stability):
                foot_score = self._calculate_foot_score(area_ratio, aspect_ratio, bbox, h, w, stability)
                foot_candidates.append((mask, foot_score, mask_data))
                
            elif self._is_reference_candidate(area_ratio, aspect_ratio, bbox, h, w):
                ref_score = self._calculate_reference_score(area_ratio, aspect_ratio, bbox, h, w)
                reference_candidates.append((mask, ref_score, mask_data))
        
        # Sélectionner les meilleurs candidats
        best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        best_ref = max(reference_candidates, key=lambda x: x[1]) if reference_candidates else None
        
        if best_foot:
            foot_mask = self._clean_foot_mask(best_foot[0])
            ref_mask = best_ref[0] if best_ref else None
            confidence = min(100, best_foot[1])
            
            return foot_mask, ref_mask, confidence
        
        return None, None, 0
    
    def _is_foot_candidate(self, area_ratio, aspect_ratio, bbox, h, w, stability):
        """Vérifie si un masque peut être un pied"""
        # Critères pour un pied
        return (
            0.08 <= area_ratio <= 0.45 and      # 8-45% de l'image
            1.8 <= aspect_ratio <= 4.5 and      # Forme allongée
            stability > 0.85 and                 # Segmentation stable
            bbox[1] + bbox[3] > h * 0.3          # Pas trop en haut
        )
    
    def _is_reference_candidate(self, area_ratio, aspect_ratio, bbox, h, w):
        """Vérifie si un masque peut être une référence (A4/carte)"""
        return (
            0.15 <= area_ratio <= 0.70 and      # 15-70% de l'image  
            1.2 <= aspect_ratio <= 2.0 and      # Forme rectangulaire
            bbox[2] * bbox[3] > 50000            # Taille minimum
        )
    
    def _calculate_foot_score(self, area_ratio, aspect_ratio, bbox, h, w, stability):
        """Calcule un score de confiance pour un candidat pied"""
        score = 0
        
        # Score basé sur l'aire (optimal 15-30%)
        if 0.15 <= area_ratio <= 0.30:
            score += 30
        elif 0.08 <= area_ratio <= 0.45:
            score += 20
        
        # Score basé sur l'aspect ratio (optimal 2.2-3.2)
        if 2.2 <= aspect_ratio <= 3.2:
            score += 25
        elif 1.8 <= aspect_ratio <= 4.0:
            score += 15
        
        # Score basé sur la position (centre-bas préférable)
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        if 0.3 <= center_x/w <= 0.7:  # Centré horizontalement
            score += 15
        if center_y/h >= 0.4:          # Dans la moitié basse
            score += 15
        
        # Score de stabilité
        score += stability * 15
        
        return score
    
    def _calculate_reference_score(self, area_ratio, aspect_ratio, bbox, h, w):
        """Calcule un score pour un candidat référence"""
        score = 0
        
        # Préférences pour références
        if 0.20 <= area_ratio <= 0.50:  # Taille modérée
            score += 20
        
        if 1.3 <= aspect_ratio <= 1.7:  # Rectangle standard
            score += 20
        
        # Position: éviter le centre (où est le pied)
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        if center_x/w < 0.3 or center_x/w > 0.7:  # Sur les côtés
            score += 15
        
        return score
    
    def _clean_foot_mask(self, mask):
        """Nettoie et optimise le masque du pied"""
        # Convertir en uint8
        clean_mask = (mask * 255).astype(np.uint8)
        
        # Opérations morphologiques spécifiques aux pieds
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Fermeture pour connecter les orteils au pied
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Ouverture légère pour supprimer le bruit
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Garder seulement le plus grand composant (le pied principal)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask.fill(0)
            cv2.fillPoly(clean_mask, [largest_contour], 255)
        
        return clean_mask
    
    def _kmeans_fallback(self, image):
        """Fallback K-means si SAM échoue"""
        try:
            # Utiliser le pipeline existant
            preprocessed = preprocess(image)
            clustered = kMeans_cluster(preprocessed)
            edges = edgeDetection(clustered)
            return edges
        except Exception as e:
            print(f"❌ K-means fallback échoué: {e}")
            return None

# ===== FONCTIONS PRINCIPALES POUR L'APPLICATION =====

def process_foot_for_mobile_app(image_path, save_debug=False):
    """
    Fonction principale pour application mobile podologue
    
    Args:
        image_path: Chemin vers l'image du pied
        save_debug: Sauvegarder les images de debug
    
    Returns:
        dict: Mesures complètes + métadonnées
    """
    print(f"\n📱 TRAITEMENT MOBILE PODOLOGUE: {os.path.basename(image_path)}")
    
    # Charger l'image
    try:
        from skimage.io import imread
        original_image = imread(image_path)
        print(f"✅ Image: {original_image.shape[1]}x{original_image.shape[0]}px")
    except Exception as e:
        return {'error': f"Erreur chargement image: {e}"}
    
    # Initialiser SAM mobile
    sam_segmenter = MobileSAMFootSegmenter()
    
    # Segmentation intelligente
    segmentation_result = sam_segmenter.segment_foot_smart(original_image)
    
    if segmentation_result['foot_mask'] is None:
        return {'error': "Impossible de segmenter le pied"}
    
    # Convertir le masque en format compatible avec le pipeline existant
    foot_mask = segmentation_result['foot_mask']
    
    # Traitement avec le pipeline existant adapté
    try:
        measurements = process_with_mask(original_image, foot_mask, segmentation_result)
        
        # Ajouter métadonnées mobiles
        measurements.update({
            'segmentation_method': segmentation_result['method_used'],
            'confidence': segmentation_result['confidence'],
            'processing_timestamp': datetime.now().isoformat(),
            'image_dimensions': f"{original_image.shape[1]}x{original_image.shape[0]}",
            'mobile_optimized': True
        })
        
        # Sauvegarder debug si demandé
        if save_debug:
            save_debug_images(original_image, foot_mask, measurements, image_path)
        
        print(f"✅ Traitement terminé ({segmentation_result['method_used']})")
        return measurements
        
    except Exception as e:
        return {'error': f"Erreur traitement: {e}"}

def process_with_mask(original_image, foot_mask, segmentation_result):
    """
    Traite l'image en utilisant le masque SAM au lieu du pipeline K-means
    """
    # Créer les contours à partir du masque
    contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        raise Exception("Aucun contour trouvé dans le masque")
    
    # Prendre le plus grand contour (le pied)
    main_contour = max(contours, key=cv2.contourArea)
    
    # Créer des structures compatibles avec le code existant
    boundRect = cv2.boundingRect(main_contour)
    
    # Simuler la structure attendue par le code existant
    fboundRect = [None, None, boundRect]  # Index 2 pour le pied
    fcnt = [None, None, main_contour]     # Index 2 pour le contour du pied
    
    # Créer une image croppée autour du pied
    x, y, w, h = boundRect
    margin = 20  # Marge autour du pied
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(original_image.shape[1], x + w + margin)
    y2 = min(original_image.shape[0], y + h + margin)
    
    pcropedImg = original_image[y1:y2, x1:x2]
    
    # Calculer les mesures avec les fonctions existantes
    measurements = calcAdvancedFootMeasures(pcropedImg, fboundRect, fcnt)
    
    # Validation spécifique mobile
    if measurements['length'] < 10 or measurements['length'] > 40:
        print(f"⚠️  Mesures inhabituelles: {measurements['length']:.1f}cm")
    
    return measurements

def save_debug_images(original_image, foot_mask, measurements, image_path):
    """Sauvegarde les images de debug pour l'application mobile"""
    debug_dir = f"output/mobile_debug/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(debug_dir, exist_ok=True)
    
    # Image originale
    cv2.imwrite(f"{debug_dir}/01_original.jpg", cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    
    # Masque du pied
    cv2.imwrite(f"{debug_dir}/02_foot_mask.jpg", foot_mask)
    
    # Superposition
    overlay = original_image.copy()
    overlay[foot_mask > 0] = overlay[foot_mask > 0] * 0.7 + np.array([0, 255, 0]) * 0.3
    cv2.imwrite(f"{debug_dir}/03_overlay.jpg", cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
    
    # Rapport texte
    with open(f"{debug_dir}/04_measurements.txt", 'w') as f:
        f.write(f"MESURES MOBILES - {os.path.basename(image_path)}\n")
        f.write("="*40 + "\n")
        f.write(f"Méthode: {measurements.get('segmentation_method')}\n")
        f.write(f"Confiance: {measurements.get('confidence')}%\n")
        f.write(f"Longueur: {measurements.get('length', 0):.2f} cm\n")
        f.write(f"Largeur: {measurements.get('width', 0):.2f} cm\n")
        f.write(f"Ratio: {measurements.get('length_width_ratio', 0):.2f}\n")
    
    print(f"🔍 Debug sauvegardé: {debug_dir}")

def quick_foot_measurement(image_path):
    """
    Version ultra-rapide pour application mobile
    Retourne seulement longueur et largeur
    """
    measurements = process_foot_for_mobile_app(image_path, save_debug=False)
    
    if 'error' in measurements:
        return measurements
    
    return {
        'length_cm': round(measurements.get('length', 0), 1),
        'width_cm': round(measurements.get('width', 0), 1),
        'ratio': round(measurements.get('length_width_ratio', 0), 2),
        'confidence': measurements.get('confidence', 0),
        'method': measurements.get('segmentation_method', 'Unknown')
    }

# ===== INTERFACE MOBILE =====

def mobile_app_interface():
    """Interface simplifiée pour intégration mobile"""
    print("""
📱 SAM PODOLOGUE - INTERFACE MOBILE

🚀 UTILISATION RAPIDE:
from mobile_sam_podiatry import quick_foot_measurement

# Mesure rapide
result = quick_foot_measurement('chemin/vers/photo_pied.jpg')
print(f"Longueur: {result['length_cm']} cm")
print(f"Largeur: {result['width_cm']} cm")

🔧 UTILISATION AVANCÉE:
result = process_foot_for_mobile_app('photo.jpg', save_debug=True)

📊 RÉPONSE TYPE:
{
    'length_cm': 25.3,
    'width_cm': 9.8, 
    'ratio': 2.58,
    'confidence': 87,
    'method': 'SAM'
}
""")

if __name__ == "__main__":
    mobile_app_interface()
    
    # Test avec une image
    test_image = "data/tt.jpg"  # Votre image de test
    if os.path.exists(test_image):
        print(f"\n🧪 TEST avec {test_image}:")
        result = quick_foot_measurement(test_image)
        print("Résultat:", result)
    else:
        print(f"\n⚠️  Image de test non trouvée: {test_image}")