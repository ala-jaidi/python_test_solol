# mobile_sam_podiatry.py - Pipeline unifié SAM pour application mobile podologue
# Segmentation pied + carte de crédit + correction perspective + mesures

import cv2
import numpy as np
import os
import torch
from datetime import datetime
from scipy.spatial import distance
import utils

# SAM imports avec gestion d'erreur
try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM non disponible - installer avec: pip install segment-anything")

# Constantes carte de crédit standard (ISO/IEC 7810 ID-1)
CREDIT_CARD_WIDTH_MM = 85.6
CREDIT_CARD_HEIGHT_MM = 53.98
CARD_ASPECT_RATIO = CREDIT_CARD_WIDTH_MM / CREDIT_CARD_HEIGHT_MM

class MobileSAMPodiatryPipeline:
    """Pipeline unifié: SAM segmente pied + carte, corrige perspective, mesure"""
    
    def __init__(self, model_type="vit_b", device=None):
        self.initialized = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        if not SAM_AVAILABLE:
            return
        
        # Charger SAM
        checkpoint_path = self._get_or_download_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                self.sam.to(device=self.device)
                
                # Predictor pour segmentation interactive
                self.predictor = SamPredictor(self.sam)
                
                # Générateur automatique optimisé
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=20,         # Plus de points pour détecter la carte
                    pred_iou_thresh=0.88,
                    stability_score_thresh=0.92,
                    crop_n_layers=0,
                    min_mask_region_area=1000,
                )
                
                self.initialized = True
                print(f"✅ Pipeline SAM unifié initialisé ({self.device})")
                
            except Exception as e:
                print(f"❌ Erreur initialisation SAM: {e}")
    
    def _get_or_download_checkpoint(self):
        """Télécharge le checkpoint SAM si nécessaire"""
        checkpoint_dir = "sam_mobile"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = f"{checkpoint_dir}/sam_{self.model_type}_mobile.pth"

        
        if not os.path.exists(checkpoint_path):
            print(f"📥 Téléchargement SAM {self.model_type}...")
            
            urls = {
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            }
            
            if self.model_type not in urls:
                print(f"❌ Modèle {self.model_type} non supporté")
                return None
            
            try:
                import requests
                response = requests.get(urls[self.model_type], stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(checkpoint_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r📥 Téléchargement: {percent:.1f}%", end='')
                
                print(f"\n✅ SAM téléchargé: {checkpoint_path}")
                
            except Exception as e:
                print(f"❌ Erreur téléchargement: {e}")
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                return None
        
        return checkpoint_path
    
    def process_foot_image(self, image_path, debug=False):
        """
        Pipeline simplifié : segmentation SAM + ratio direct + mesures SANS warp.
        """
        print(f"\n📱 PIPELINE SANS WARP: {os.path.basename(image_path)}")

        # 1. Charger l'image
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f"Impossible de charger l'image: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print(f"📸 Image: {w}x{h}px")

        if not self.initialized:
            return {'error': "SAM non initialisé"}

        # 2. Segmentation SAM
        print("🤖 Segmentation SAM en cours...")
        masks = self.mask_generator.generate(image_rgb)

        if not masks:
            return {'error': "Aucun masque généré par SAM"}

        # 3. Identifier pied et carte
        foot_mask, card_mask, _ = self._identify_foot_and_card(masks, image_rgb)

        if foot_mask is None:
            return {'error': "Pied non détecté"}
        if card_mask is None:
            return {'error': "Carte non détectée"}

        # 4. Calculer le ratio directement sur la carte
        print("📏 Calibration directe via carte...")
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'error': "Contour carte non trouvé"}
        card_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(card_contour)
        card_px = max(rect[1])  # largeur ou hauteur en pixels
        ratio_px_mm = card_px / CREDIT_CARD_WIDTH_MM
        print(f"✅ Largeur carte: {card_px:.1f}px = {CREDIT_CARD_WIDTH_MM}mm → ratio: {ratio_px_mm:.3f} px/mm")

        # 5. Mesurer le pied dans l'image brute
        measurements = self._measure_foot(image_rgb, foot_mask, ratio_px_mm)

        # 6. Ajouter métadonnées
        measurements.update({
            'image_path': image_path,
            'original_dimensions': f"{w}x{h}",
            'perspective_corrected': False,
            'card_detected': True,
            'processing_timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_confidence(foot_mask, card_mask)
        })

        # 7. Debug si demandé
        if debug:
            self._save_debug_images(
                image_rgb, foot_mask, card_mask, None,
                image_rgb, foot_mask, measurements
            )

        print(f"✅ Mesures terminées: {measurements['length_cm']:.1f}cm x {measurements['width_cm']:.1f}cm")
        return measurements
    
    
    def _identify_foot_and_card(self, masks, image):
        """Identifie le pied et la carte de crédit parmi les masques SAM"""
        h, w = image.shape[:2]
        image_area = h * w
        
        foot_candidates = []
        card_candidates = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']  # x, y, w, h
            
            # Calculer propriétés géométriques
            area_ratio = area / image_area
            bbox_w, bbox_h = bbox[2], bbox[3]
            aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if min(bbox_w, bbox_h) > 0 else 0
            
            # Score pour pied (forme allongée, taille moyenne)
            if 0.08 <= area_ratio <= 0.45 and 1.8 <= aspect_ratio <= 4.5:
                foot_score = self._score_foot_candidate(mask_data, h, w)
                foot_candidates.append((mask, foot_score, mask_data))
            
            # Score pour carte (rectangle, bon aspect ratio)
            if 0.005 <= area_ratio <= 0.15:  # Carte plus petite que le pied
                card_score = self._score_card_candidate(mask_data, h, w)
                if card_score > 0:
                    card_candidates.append((mask, card_score, mask_data))
        
        # Sélectionner les meilleurs candidats
        best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        best_card = max(card_candidates, key=lambda x: x[1]) if card_candidates else None
        
        foot_mask = None
        card_mask = None
        card_corners = None
        
        if best_foot:
            foot_mask = (best_foot[0] * 255).astype(np.uint8)
            foot_mask = self._clean_mask(foot_mask)
        
        if best_card:
            card_mask = (best_card[0] * 255).astype(np.uint8)
            card_corners = self._find_card_corners(card_mask)
        
        return foot_mask, card_mask, card_corners
    
    def _score_foot_candidate(self, mask_data, h, w):
        """Score un candidat pied"""
        score = 0
        area_ratio = mask_data['area'] / (h * w)
        bbox = mask_data['bbox']
        aspect_ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
        
        # Taille idéale pour un pied
        if 0.15 <= area_ratio <= 0.30:
            score += 30
        elif 0.08 <= area_ratio <= 0.45:
            score += 15
        
        # Forme allongée
        if 2.2 <= aspect_ratio <= 3.2:
            score += 25
        elif 1.8 <= aspect_ratio <= 4.0:
            score += 10
        
        # Position (plutôt centré et pas trop haut)
        center_y = bbox[1] + bbox[3]/2
        if center_y > h * 0.3:
            score += 15
        
        # Stabilité SAM
        score += mask_data['stability_score'] * 30
        
        return score
    
    def _score_card_candidate(self, mask_data, h, w):
        """Score un candidat carte de crédit"""
        bbox = mask_data['bbox']
        bbox_w, bbox_h = bbox[2], bbox[3]
        
        if bbox_w == 0 or bbox_h == 0:
            return 0
        
        # Calculer l'aspect ratio du candidat
        candidate_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
        
        # Comparer avec le ratio standard d'une carte
        ratio_diff = abs(candidate_ratio - CARD_ASPECT_RATIO)
        
        # Si trop éloigné du ratio carte, score 0
        if ratio_diff > 0.3:
            return 0
        
        score = 50  # Score de base
        
        # Bonus pour ratio proche
        score += (1 - ratio_diff) * 30
        
        # Bonus pour taille raisonnable (3-10% de l'image)
        area_ratio = mask_data['area'] / (h * w)
        if 0.03 <= area_ratio <= 0.10:
            score += 20
        
        # Bonus pour forme rectangulaire (convexité)
        mask = mask_data['segmentation']
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            cnt = contours[0]
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            cnt_area = cv2.contourArea(cnt)
            
            if hull_area > 0:
                convexity = cnt_area / hull_area
                if convexity > 0.95:  # Très convexe = rectangulaire
                    score += 20
        
        return score
    
    def _find_card_corners(self, card_mask):
        """Trouve les 4 coins de la carte de crédit"""
        # Trouver le contour principal
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Prendre le plus grand contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Approximer à un polygone
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)
        
        # Si on a 4 points, parfait
        if len(approx) == 4:
            corners = approx.reshape(-1, 2)
        else:
            # Sinon, utiliser le rectangle minimum
            rect = cv2.minAreaRect(main_contour)
            corners = cv2.boxPoints(rect)
            corners = np.int0(corners)
        
        # Ordonner les coins (haut-gauche, haut-droit, bas-droit, bas-gauche)
        corners = self._order_corners(corners)
        
        return corners
    
    def _order_corners(self, pts):
        """Ordonne les 4 coins dans l'ordre standard"""
        # Trier par somme (x+y) pour trouver haut-gauche et bas-droit
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]     # Haut-gauche
        ordered[2] = pts[np.argmax(s)]     # Bas-droit
        ordered[1] = pts[np.argmin(diff)]  # Haut-droit
        ordered[3] = pts[np.argmax(diff)]  # Bas-gauche
        
        return ordered
    
    def _correct_perspective(self, image, card_corners):
        """Corrige la perspective en utilisant la carte comme référence"""
        # Dimensions cibles pour la carte (en pixels)
        # On garde un ratio correct
        target_width = 400  # pixels
        target_height = int(target_width / CARD_ASPECT_RATIO)
        
        # Points de destination pour la carte
        dst_corners = np.array([
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1]
        ], dtype=np.float32)
        
        # Calculer la transformation perspective
        transform_matrix = cv2.getPerspectiveTransform(
            card_corners.astype(np.float32), 
            dst_corners
        )
        
        # Calculer la taille finale de l'image
        # On veut garder tout le contenu visible
        h, w = image.shape[:2]
        corners_img = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1]
        ], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transformer les coins de l'image
        transformed_corners = cv2.perspectiveTransform(corners_img, transform_matrix)
        
        # Trouver la boîte englobante
        x_min = int(transformed_corners[:, :, 0].min())
        x_max = int(transformed_corners[:, :, 0].max())
        y_min = int(transformed_corners[:, :, 1].min())
        y_max = int(transformed_corners[:, :, 1].max())
        
        # Ajuster la transformation pour tout garder visible
        translation = np.array([
            [1, 0, -x_min],
            [0, 1, -y_min],
            [0, 0, 1]
        ])
        
        adjusted_transform = translation @ transform_matrix
        output_width = x_max - x_min
        output_height = y_max - y_min
        
        # Appliquer la transformation
        warped = cv2.warpPerspective(
            image, 
            adjusted_transform, 
            (output_width, output_height),
            flags=cv2.INTER_LINEAR
        )
        
        return warped, adjusted_transform
    
    def _calculate_ratio_from_card(self, warped_image):
        """Calcule le ratio px/mm à partir de la carte dans l'image corrigée"""
        # Comme on a défini target_width = 400 pixels pour la carte après correction
        # et que la carte fait CREDIT_CARD_WIDTH_MM millimètres
        # Le ratio est simplement :
        target_card_width_px = 400  # Défini dans _correct_perspective
        ratio_px_mm = target_card_width_px / CREDIT_CARD_WIDTH_MM
        
        print(f"✅ Carte normalisée: {target_card_width_px}px = {CREDIT_CARD_WIDTH_MM}mm")
        print(f"📏 Ratio: {ratio_px_mm:.3f} px/mm")
        
        return ratio_px_mm
    
    def _measure_foot(self, image, foot_mask, ratio_px_mm):
        """Mesure le pied sur l'image corrigée"""
        # Trouver le contour du pied
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {'error': 'Contour du pied non trouvé'}
        
        # Prendre le plus grand contour
        foot_contour = max(contours, key=cv2.contourArea)
        
        # Rectangle englobant
        x, y, w, h = cv2.boundingRect(foot_contour)
        
        # Mesures de base
        length_px = h  # Hauteur = longueur du pied
        width_px = w   # Largeur maximale
        
        # Convertir en cm
        length_cm = (length_px / ratio_px_mm) / 10
        width_cm = (width_px / ratio_px_mm) / 10
        
        # Mesures avancées
        # Trouver la largeur réelle (pas juste le rectangle)
        real_width_px = self._find_real_width(foot_mask, foot_contour)
        real_width_cm = (real_width_px / ratio_px_mm) / 10
        
        # Points caractéristiques
        heel_point, toe_point = self._find_heel_and_toe(foot_contour)
        
        # Distance réelle talon-orteil
        real_length_px = distance.euclidean(heel_point, toe_point)
        real_length_cm = (real_length_px / ratio_px_mm) / 10
        
        # Aire du pied
        foot_area_px = cv2.contourArea(foot_contour)
        foot_area_cm2 = (foot_area_px / (ratio_px_mm ** 2)) / 100
        
        # Périmètre
        perimeter_px = cv2.arcLength(foot_contour, True)
        perimeter_cm = (perimeter_px / ratio_px_mm) / 10

        measurements = {
            'length_cm': round(real_length_cm, 2),
            'width_cm': round(real_width_cm, 2),
            'length_width_ratio': round(real_length_cm / real_width_cm, 2),
            'area_cm2': round(foot_area_cm2, 2),
            'perimeter_cm': round(perimeter_cm, 2),
            'heel_position': heel_point.tolist(),
            'toe_position': toe_point.tolist(),
            'bounding_box': {
                'x': x, 'y': y, 'w': w, 'h': h
            },
            'ratio_px_mm': round(ratio_px_mm, 3)
        }

        # Analyse supplémentaire
        measurements.update(self._analyze_toes(foot_contour, heel_point, toe_point, ratio_px_mm))
        measurements.update(utils.estimateInstepHeightRobust(foot_contour, ratio_px_mm, ratio_px_mm))
        measurements.update(utils.analyzeArchSupportRobust(foot_contour, ratio_px_mm, ratio_px_mm))

        return measurements
    
    def _find_real_width(self, foot_mask, contour):
        """Trouve la largeur réelle maximale du pied"""
        # Créer un masque vide
        h, w = foot_mask.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        
        # Scanner horizontalement pour trouver la largeur max
        max_width = 0
        
        for y in range(h):
            row = mask[y, :]
            if np.any(row):
                indices = np.where(row)[0]
                width = indices[-1] - indices[0]
                max_width = max(max_width, width)
        
        return max_width
    
    def _find_heel_and_toe(self, contour):
        """Trouve les points talon et orteil"""
        # Point le plus bas = talon
        heel_idx = contour[:, :, 1].argmax()
        heel_point = contour[heel_idx, 0]
        
        # Point le plus haut = orteil
        toe_idx = contour[:, :, 1].argmin()
        toe_point = contour[toe_idx, 0]

        return heel_point, toe_point

    def _analyze_toes(self, contour, heel_point, toe_point, ratio_px_mm):
        """Estimate distances from heel to big and little toes."""
        pts = contour[:, 0, :]
        toe_y = pts[:, 1].min()
        region = pts[pts[:, 1] <= toe_y + 5]
        if len(region) == 0:
            region = pts[pts[:, 1] <= toe_y + 10]
        if len(region) == 0:
            return {
                "bigtoe_to_heel_cm": 0.0,
                "littletoe_to_heel_cm": 0.0,
            }
        bigtoe = region[region[:, 0].argmax()]
        littletoe = region[region[:, 0].argmin()]
        big_d = distance.euclidean(bigtoe, heel_point)
        little_d = distance.euclidean(littletoe, heel_point)
        return {
            "bigtoe_to_heel_cm": round((big_d / ratio_px_mm) / 10, 2),
            "littletoe_to_heel_cm": round((little_d / ratio_px_mm) / 10, 2),
        }
    
    def _clean_mask(self, mask):
        """Nettoie le masque (morphologie)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Fermeture pour connecter les parties
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Ouverture pour supprimer le bruit
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Garder seulement le plus grand composant
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask.fill(0)
            cv2.fillPoly(mask, [largest], 255)
        
        return mask
    
    def _calculate_confidence(self, foot_mask, card_mask):
        """Calcule un score de confiance global"""
        confidence = 50  # Base
        
        if foot_mask is not None:
            confidence += 25
        
        if card_mask is not None:
            confidence += 25
        
        return min(confidence, 100)
    
    def _save_debug_images(self, original, foot_mask, card_mask, card_corners,
                          warped, warped_foot_mask, measurements):
        """Sauvegarde les images de debug"""
        debug_dir = f"output/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_dir, exist_ok=True)
        
        # 1. Image originale avec masques
        vis1 = original.copy()
        if foot_mask is not None:
            foot_overlay = np.zeros_like(vis1)
            foot_overlay[foot_mask > 0] = [0, 255, 0]
            vis1 = cv2.addWeighted(vis1, 0.7, foot_overlay, 0.3, 0)
        
        if card_mask is not None:
            card_overlay = np.zeros_like(vis1)
            card_overlay[card_mask > 0] = [255, 0, 0]
            vis1 = cv2.addWeighted(vis1, 0.7, card_overlay, 0.3, 0)
        
        if card_corners is not None:
            cv2.polylines(vis1, [card_corners.astype(int)], True, (255, 255, 0), 3)
            for i, corner in enumerate(card_corners):
                cv2.circle(vis1, tuple(corner.astype(int)), 10, (255, 0, 255), -1)
                cv2.putText(vis1, str(i), tuple(corner.astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imwrite(f"{debug_dir}/01_segmentation.jpg", cv2.cvtColor(vis1, cv2.COLOR_RGB2BGR))
        
        # 2. Image corrigée
        cv2.imwrite(f"{debug_dir}/02_perspective_corrected.jpg", 
                   cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
        
        # 3. Masque pied corrigé
        cv2.imwrite(f"{debug_dir}/03_foot_mask_corrected.jpg", warped_foot_mask)
        
        # 4. Mesures visualisées
        vis_measures = warped.copy()
        contours, _ = cv2.findContours(warped_foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cv2.drawContours(vis_measures, contours, -1, (0, 255, 0), 2)
            cv2.drawContours(vis1, contours, -1, (0, 255, 255), 2)

            # Dessiner les points caractéristiques
            if 'heel_position' in measurements:
                heel = measurements['heel_position']
                toe = measurements['toe_position']
                cv2.circle(vis_measures, tuple(heel), 10, (255, 0, 0), -1)
                cv2.circle(vis_measures, tuple(toe), 10, (0, 0, 255), -1)
                cv2.line(vis_measures, tuple(heel), tuple(toe), (255, 255, 0), 2)
        
        cv2.imwrite(f"{debug_dir}/04_measurements.jpg", 
                   cv2.cvtColor(vis_measures, cv2.COLOR_RGB2BGR))
        
        # 5. Rapport texte
        with open(f"{debug_dir}/05_report.txt", 'w', encoding='utf-8') as f:
            f.write("RAPPORT DE MESURES PODIATRIQUES\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {measurements['processing_timestamp']}\n")
            f.write(f"Image: {measurements['image_path']}\n")
            f.write(f"Dimensions originales: {measurements['original_dimensions']}\n")
            f.write(f"Perspective corrigée: {measurements['perspective_corrected']}\n")
            f.write(f"Carte détectée: {measurements['card_detected']}\n")
            f.write(f"Confiance: {measurements['confidence']}%\n\n")
            
            f.write("MESURES DU PIED:\n")
            f.write(f"- Longueur: {measurements['length_cm']} cm\n")
            f.write(f"- Largeur: {measurements['width_cm']} cm\n")
            f.write(f"- Ratio L/l: {measurements['length_width_ratio']}\n")
            f.write(f"- Surface: {measurements['area_cm2']} cm²\n")
            f.write(f"- Périmètre: {measurements['perimeter_cm']} cm\n")
            f.write(f"\nRatio px/mm: {measurements['ratio_px_mm']}\n")
        
        print(f"📁 Debug sauvegardé dans: {debug_dir}")


# ===== FONCTIONS UTILITAIRES =====

def quick_measure(image_path):
    """
    Mesure rapide pour intégration mobile
    
    Args:
        image_path: Chemin vers l'image
    
    Returns:
        dict: Mesures essentielles
    """
    pipeline = MobileSAMPodiatryPipeline()
    
    if not pipeline.initialized:
        return {'error': 'Pipeline non initialisé - SAM requis'}
    
    result = pipeline.process_foot_image(image_path, debug=False)
    
    if 'error' in result:
        return result
    
    # Retourner seulement l'essentiel pour mobile
    return {
        'success': True,
        'length_cm': result['length_cm'],
        'width_cm': result['width_cm'],
        'ratio': result['length_width_ratio'],
        'confidence': result['confidence']
    }


def batch_process_folder(folder_path, output_csv=None):
    """
    Traite tous les images d'un dossier
    
    Args:
        folder_path: Dossier contenant les images
        output_csv: Fichier CSV de sortie (optionnel)
    """
    import glob
    import pandas as pd
    
    pipeline = MobileSAMPodiatryPipeline()
    
    if not pipeline.initialized:
        print("❌ Pipeline non initialisé")
        return
    
    # Trouver toutes les images
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans {folder_path}")
        return
    
    print(f"📁 {len(image_files)} images à traiter")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")
        
        try:
            measurements = pipeline.process_foot_image(image_path, debug=True)
            
            if 'error' not in measurements:
                results.append({
                    'filename': os.path.basename(image_path),
                    'length_cm': measurements['length_cm'],
                    'width_cm': measurements['width_cm'],
                    'ratio': measurements['length_width_ratio'],
                    'area_cm2': measurements['area_cm2'],
                    'confidence': measurements['confidence']
                })
                print(f"✅ OK: {measurements['length_cm']}cm x {measurements['width_cm']}cm")
            else:
                print(f"❌ Erreur: {measurements['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    # Sauvegarder les résultats
    if results:
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n📊 Résultats sauvegardés: {output_csv}")
        
        # Afficher statistiques
        print("\n📊 STATISTIQUES:")
        print(f"Images traitées: {len(results)}/{len(image_files)}")
        print(f"Longueur moyenne: {df['length_cm'].mean():.1f} cm")
        print(f"Largeur moyenne: {df['width_cm'].mean():.1f} cm")
        print(f"Confiance moyenne: {df['confidence'].mean():.0f}%")


def process_multiview(images, debug=False):
    """Process multiple views of the same foot and combine metrics.

    Args:
        images (FaceImages): container with image paths
        debug (bool): save debug images for each view
    """
    pipeline = MobileSAMPodiatryPipeline()

    if not pipeline.initialized:
        return {'error': 'Pipeline non initialisé - SAM requis'}

    view_results = {}
    for view in ['top', 'left', 'right', 'front', 'back']:
        path = getattr(images, view)
        if not os.path.exists(path):
            view_results[view] = {'error': 'Fichier manquant'}
            continue
        view_results[view] = pipeline.process_foot_image(path, debug=debug)

    lengths = [r['length_cm'] for r in view_results.values() if 'length_cm' in r]
    widths = [r['width_cm'] for r in view_results.values() if 'width_cm' in r]
    insteps = [r['instep_height_estimate_cm'] for r in view_results.values() if 'instep_height_estimate_cm' in r]
    archs = [r['arch_type'] for r in view_results.values() if 'arch_type' in r]

    combined = {}
    if lengths:
        combined['length_cm'] = round(float(np.mean(lengths)), 2)
    if widths:
        combined['width_cm'] = round(float(np.mean(widths)), 2)
    if insteps:
        combined['instep_height_cm'] = round(float(np.mean(insteps)), 2)
    if archs:
        combined['arch_type'] = max(set(archs), key=archs.count)

    combined['views'] = view_results
    return combined


def validate_setup():
    """Vérifie que tout est correctement installé"""
    print("🔍 Vérification de l'installation...\n")
    
    # Check Python version
    import sys
    print(f"✓ Python {sys.version.split()[0]}")
    
    # Check packages
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'segment_anything': 'segment-anything'
    }
    
    missing = []
    
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"✓ {package} installé")
        except ImportError:
            print(f"✗ {package} manquant")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Installer les packages manquants:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible ({torch.cuda.get_device_name(0)})")
        else:
            print("ℹ️  CUDA non disponible - utilisation CPU")
    except:
        pass
    
    print("\n✅ Installation correcte!")
    return True


# ===== INTERFACE PRINCIPALE =====

def main():
    """Interface en ligne de commande"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pipeline SAM unifié pour mesures podiatriques",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  %(prog)s image.jpg                    # Mesure une image
  %(prog)s image.jpg --debug           # Avec images de debug
  %(prog)s --batch dossier/            # Traiter un dossier
  %(prog)s --validate                  # Vérifier l'installation
        """
    )
    
    parser.add_argument('image', nargs='?', help='Image à analyser')
    parser.add_argument('--debug', action='store_true', help='Sauvegarder les images de debug')
    parser.add_argument('--batch', metavar='FOLDER', help='Traiter toutes les images d\'un dossier')
    parser.add_argument('--output', metavar='CSV', help='Fichier CSV de sortie (pour --batch)')
    parser.add_argument('--validate', action='store_true', help='Vérifier l\'installation')
    parser.add_argument('--model', choices=['vit_b', 'vit_l', 'vit_h'], default='vit_b',
                       help='Modèle SAM à utiliser (défaut: vit_b)')
    
    args = parser.parse_args()
    
    # Validation
    if args.validate:
        validate_setup()
        return
    
    # Batch processing
    if args.batch:
        batch_process_folder(args.batch, args.output)
        return
    
    # Single image
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Fichier introuvable: {args.image}")
            return
        
        # Initialiser le pipeline
        print(f"🚀 Initialisation du pipeline (modèle: {args.model})...")
        pipeline = MobileSAMPodiatryPipeline(model_type=args.model)
        
        if not pipeline.initialized:
            print("❌ Impossible d'initialiser le pipeline")
            print("Vérifiez l'installation avec: python mobile_sam_podiatry.py --validate")
            return
        
        # Traiter l'image
        result = pipeline.process_foot_image(args.image, debug=args.debug)
        
        # Afficher les résultats
        if 'error' in result:
            print(f"\n❌ Erreur: {result['error']}")
        else:
            print("\n✅ MESURES DU PIED:")
            print(f"📏 Longueur: {result['length_cm']} cm")
            print(f"📐 Largeur: {result['width_cm']} cm")
            print(f"🔢 Ratio L/l: {result['length_width_ratio']}")
            print(f"📊 Surface: {result['area_cm2']} cm²")
            print(f"🔄 Périmètre: {result['perimeter_cm']} cm")
            print(f"✨ Confiance: {result['confidence']}%")
            
            if args.debug:
                print("\n📁 Images de debug sauvegardées dans output/")
    
    else:
        # Afficher l'aide si aucun argument
        parser.print_help()
        
        print("\n📱 UTILISATION RAPIDE POUR MOBILE:")
        print("from mobile_sam_podiatry import quick_measure")
        print("result = quick_measure('photo_pied.jpg')")
        print("print(f\"Pied: {result[\'length_cm\']}cm x {result[\'width_cm\']}cm\")")


if __name__ == "__main__":
    main()