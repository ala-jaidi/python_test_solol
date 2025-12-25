# mobile_sam_podiatry.py - Pipeline SAM simplifié pour application mobile podologue
# SIDE uniquement via process_foot_image avec _find_heel_and_toe (talon=max Y, orteil=min Y)

import cv2
import numpy as np
import os
import torch
from datetime import datetime
from utils import keep_foot_only
from dxf_export import DXFExporter

# ============================================================
# A) CHARGEMENT / SAM
# ============================================================

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

# ArUco L-shaped board configuration
ARUCO_L_BOARD_SIZE_MM = 60.0
ARUCO_L_BOARD_SEPARATION_MM = 12.0
ARUCO_DICT = cv2.aruco.DICT_6X6_250


class MobileSAMPodiatryPipeline:
    """Pipeline unifié: SAM segmente pied, calibration ArUco/carte, mesures"""
    
    # ============================================================
    # A) CHARGEMENT / SAM
    # ============================================================
    
    def __init__(self, model_type="vit_b", device=None):
        """Initialise le pipeline SAM"""
        self.initialized = False
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        
        if not SAM_AVAILABLE:
            return
        
        checkpoint_path = self._get_or_download_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                self.sam.to(device=self.device)
                
                self.predictor = SamPredictor(self.sam)
                
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=20,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.92,
                    crop_n_layers=0,
                    min_mask_region_area=1200,
                    box_nms_thresh=0.7
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

    # ============================================================
    # B) CALIBRATION
    # ============================================================
    
    def _detect_aruco_l_board(self, image):
        """
        Détecte le L-board ArUco et calcule le ratio px/mm
        Returns: (ratio_px_mm, calibration_data, marker_positions) ou (None, None, None)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        corners, ids, _ = detector.detectMarkers(gray)
        
        # Tentative 2: Paramètres permissifs si échec
        if ids is None or len(ids) < 2:
            # print("⚠️ ArUco: Tentative avec paramètres permissifs...")
            parameters.adaptiveThreshWinSizeMin = 3
            parameters.adaptiveThreshWinSizeMax = 23
            parameters.adaptiveThreshWinSizeStep = 10
            parameters.minMarkerPerimeterRate = 0.03
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)

        # Tentative 3: Amélioration contraste (CLAHE) si échec
        if ids is None or len(ids) < 2:
            # print("⚠️ ArUco: Tentative avec CLAHE...")
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            gray_enhanced = clahe.apply(gray)
            corners, ids, _ = detector.detectMarkers(gray_enhanced)

        if ids is None or len(ids) < 2:
            return None, None, None
        
        marker_positions = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2]:
                marker_positions[marker_id] = corners[i][0]
        
        if 0 not in marker_positions or 1 not in marker_positions:
            print("⚠️ ArUco L-board: Markers 0 and 1 required")
            return None, None, None
        
        marker0_center = np.mean(marker_positions[0], axis=0)
        marker1_center = np.mean(marker_positions[1], axis=0)
        
        distance_px = np.linalg.norm(marker1_center - marker0_center)
        known_distance_mm = ARUCO_L_BOARD_SIZE_MM + ARUCO_L_BOARD_SEPARATION_MM
        ratio_px_mm = distance_px / known_distance_mm
        
        # Pose 3D (optionnel)
        pose_info = None
        if 2 in marker_positions:
            try:
                object_points = np.array([
                    [0, 0, 0],
                    [known_distance_mm, 0, 0],
                    [0, known_distance_mm, 0]
                ], dtype=np.float32)
                
                marker2_center = np.mean(marker_positions[2], axis=0)
                image_points = np.array([
                    marker0_center, marker1_center, marker2_center
                ], dtype=np.float32)
                
                image_size = gray.shape[::-1]
                focal_length = max(image_size)
                camera_matrix = np.array([
                    [focal_length, 0, image_size[0]/2],
                    [0, focal_length, image_size[1]/2],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                dist_coeffs = np.zeros((4, 1))
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, camera_matrix, dist_coeffs
                )
                if success:
                    pose_info = {'rvec': rvec, 'tvec': tvec, 'camera_matrix': camera_matrix}
            except:
                pass
        
        calibration_data = {
            'ratio_px_mm': ratio_px_mm,
            'marker_positions': marker_positions,
            'distance_px': distance_px,
            'known_distance_mm': known_distance_mm,
            'pose_info': pose_info,
            'board_detected': True
        }
        
        print(f"✅ ArUco L-board detected: {distance_px:.1f}px = {known_distance_mm}mm")
        print(f"📏 Ratio: {ratio_px_mm:.3f} px/mm")
        
        return ratio_px_mm, calibration_data, marker_positions

    # ============================================================
    # C) SEGMENTATION + NETTOYAGE
    # ============================================================
    
    def _identify_foot_and_card(self, masks, image):
        """Identifie le pied et la carte de crédit parmi les masques SAM"""
        h, w = image.shape[:2]
        image_area = h * w
        
        foot_candidates = []
        card_candidates = []
        
        # Collecter candidats carte
        for mask_data in masks:
            area_ratio = mask_data['area'] / image_area
            if 0.005 <= area_ratio <= 0.15:
                card_score = self._score_card_candidate(mask_data, h, w)
                if card_score > 0:
                    card_candidates.append((mask_data['segmentation'], card_score, mask_data))
        
        best_card = max(card_candidates, key=lambda x: x[1]) if card_candidates else None
        card_mask = None
        if best_card:
            card_mask = (best_card[0] * 255).astype(np.uint8)
        
        # Collecter candidats pied
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']
            
            area_ratio = area / image_area
            bbox_w, bbox_h = bbox[2], bbox[3]
            aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if min(bbox_w, bbox_h) > 0 else 0
            
            if 0.08 <= area_ratio <= 0.45 and 1.8 <= aspect_ratio <= 4.5:
                if self._is_foot_like(mask_data, h, w, card_mask=card_mask):
                    foot_score = self._score_foot_candidate(mask_data, h, w)
                    foot_candidates.append((mask, foot_score, mask_data))
        
        best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        
        # Fallback si pas de pied trouvé
        if best_foot is None:
            for mask_data in masks:
                mask = mask_data['segmentation']
                area = mask_data['area']
                bbox = mask_data['bbox']
                area_ratio = area / image_area
                bbox_w, bbox_h = bbox[2], bbox[3]
                if min(bbox_w, bbox_h) <= 0:
                    continue
                aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
                if 0.05 <= area_ratio <= 0.60 and 1.2 <= aspect_ratio <= 5.0:
                    foot_score = self._score_foot_candidate(mask_data, h, w)
                    foot_candidates.append((mask, foot_score * 0.9, mask_data))
            best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        
        foot_mask = None
        if best_foot:
            foot_mask = (best_foot[0] * 255).astype(np.uint8)
            foot_mask = self._clean_mask(foot_mask)
        
        return foot_mask, card_mask, None

    def _is_foot_like(self, mask_data, H, W, card_mask=None):
        """Vérifie si un masque ressemble à un pied"""
        m = (mask_data['segmentation'] * 255).astype(np.uint8)
        x, y, w, h = mask_data['bbox']
        
        if h / H < 0.22:
            return False
        
        touches = int(y <= 2) + int(y + h >= H - 3) + int(x <= 2) + int(x + w >= W - 3)
        if touches >= 2:
            return False
        
        ys, xs = np.where(m > 0)
        if xs.size < 20:
            return False
        pts = np.c_[xs, ys].astype(np.float32)
        cov = np.cov(pts.T)
        eigvals, _ = np.linalg.eigh(cov)
        flat = eigvals.min() / eigvals.max()
        if flat < 0.12:
            return False
        
        if card_mask is not None:
            overlap = np.logical_and(m > 0, card_mask > 0).sum() / max((m > 0).sum(), 1)
            if overlap > 0.05:
                return False
        return True
    
    def _score_foot_candidate(self, mask_data, h, w):
        """Score un candidat pied"""
        score = 0
        area_ratio = mask_data['area'] / (h * w)
        bbox = mask_data['bbox']
        aspect_ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
        
        if 0.12 <= area_ratio <= 0.35:
            score += 35
        elif 0.08 <= area_ratio <= 0.50:
            score += 20
        elif 0.05 <= area_ratio <= 0.60:
            score += 10
        
        if 1.5 <= aspect_ratio <= 3.5:
            optimal_score = 30 * (1 - abs(aspect_ratio - 2.5) / 2.0)
            score += max(optimal_score, 15)
        elif 1.2 <= aspect_ratio <= 4.5:
            score += 10
        
        center_y = bbox[1] + bbox[3]/2
        if 0.3 * h <= center_y <= 0.8 * h:
            score += 20
        elif 0.2 * h <= center_y <= 0.9 * h:
            score += 10
        
        center_x = bbox[0] + bbox[2]/2
        if 0.2 * w <= center_x <= 0.8 * w:
            score += 15
        
        stability = mask_data.get('stability_score', 0)
        score += stability * 35
        
        predicted_iou = mask_data.get('predicted_iou', 0)
        score += predicted_iou * 20
        
        return score
    
    def _score_card_candidate(self, mask_data, h, w):
        """Score un candidat carte de crédit"""
        bbox = mask_data['bbox']
        bbox_w, bbox_h = bbox[2], bbox[3]
        
        if bbox_w == 0 or bbox_h == 0:
            return 0
        
        candidate_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h)
        ratio_diff = abs(candidate_ratio - CARD_ASPECT_RATIO)
        
        if ratio_diff > 0.3:
            return 0
        
        score = 50
        score += (1 - ratio_diff) * 30
        
        area_ratio = mask_data['area'] / (h * w)
        if 0.03 <= area_ratio <= 0.10:
            score += 20
        
        return score
    
    def _clean_mask(self, mask):
        """Nettoie le masque (morphologie)"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask.fill(0)
            cv2.fillPoly(mask, [largest], 255)
        
        return mask

    # ============================================================
    # D) MESURES SPÉCIFIQUES
    # ============================================================
    
    def _find_max_width_points(self, foot_mask, contour):
        """Retourne la largeur maximale et les points gauche/droit (pour TOP view)"""
        h, w = foot_mask.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        max_width = 0
        left_pt = (0, 0)
        right_pt = (0, 0)
        for y in range(h):
            indices = np.where(mask[y, :] > 0)[0]
            if len(indices) > 0:
                width = indices[-1] - indices[0]
                if width > max_width:
                    max_width = width
                    left_pt = (int(indices[0]), int(y))
                    right_pt = (int(indices[-1]), int(y))

        return max_width, left_pt, right_pt
    
    def _find_heel_and_toe(self, contour, foot_side="right"):
        """
        Trouve les points talon et orteil (Vue Profil Horizontal uniquement)
        Hypothèse (Vue Interne):
        - Pied DROIT : Talon à GAUCHE (Min X), Orteils à DROITE (Max X)
        - Pied GAUCHE : Orteils à GAUCHE (Min X), Talon à DROITE (Max X)
        """
        pts = contour[:, 0, :]
        
        # Points extrêmes sur l'axe X
        min_x_point = pts[pts[:, 0].argmin()] # Point le plus à gauche
        max_x_point = pts[pts[:, 0].argmax()] # Point le plus à droite
        
        if foot_side.lower() == "left":
            # Pied GAUCHE : Talon à Droite, Orteil à Gauche
            heel_point = max_x_point
            toe_point = min_x_point
            print(f"🔍 Profil Pied GAUCHE: Heel(max X):{heel_point}, Toe(min X):{toe_point}")
        else:
            # Pied DROIT (défaut) : Talon à Gauche, Orteil à Droite
            heel_point = min_x_point
            toe_point = max_x_point
            print(f"🔍 Profil Pied DROIT: Heel(min X):{heel_point}, Toe(max X):{toe_point}")
        
        return heel_point, toe_point
    
    def _measure_side_view_data(self, foot_mask, ratio_px_mm, foot_side="right"):
        """Calcul de la LONGUEUR (Talon-Orteil) pour Side View"""
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'error': 'Contour du pied non trouvé'}
        foot_contour = max(contours, key=cv2.contourArea)
        
        # Talon / Orteil (avec prise en compte du côté)
        heel_point, toe_point = self._find_heel_and_toe(foot_contour, foot_side)
        
        # Distance Euclidienne
        real_length_px = np.linalg.norm(heel_point - toe_point)
        length_cm = (real_length_px / ratio_px_mm) / 10
        
        # Export DXF
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dxf_filename = f"side_contour_{timestamp}.dxf"
        dxf_path = DXFExporter.export_contour_to_dxf(
            foot_contour, ratio_px_mm, "output", dxf_filename
        )
        
        return {
            'view': 'side',
            'length_cm': round(length_cm, 2),
            'heel_point': heel_point.tolist(),
            'toe_point': toe_point.tolist(),
            'ratio_px_mm': round(ratio_px_mm, 3),
            'dxf_path': dxf_path
        }

    def _measure_top_view_data(self, foot_mask, ratio_px_mm, foot_side="right"):
        """Calcul de la LARGEUR (Gauche-Droite) pour Top View"""
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'error': 'Contour du pied non trouvé'}
        foot_contour = max(contours, key=cv2.contourArea)
        
        # Largeur Max
        width_px, left_pt, right_pt = self._find_max_width_points(foot_mask, foot_contour)
        width_cm = (width_px / ratio_px_mm) / 10
        
        # Export DXF
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dxf_filename = f"top_contour_{foot_side}_{timestamp}.dxf"
        dxf_path = DXFExporter.export_contour_to_dxf(
            foot_contour, ratio_px_mm, "output", dxf_filename
        )
        
        return {
            'view': 'top',
            'width_cm': round(width_cm, 2),
            'left_point': list(left_pt),
            'right_point': list(right_pt),
            'ratio_px_mm': round(ratio_px_mm, 3),
            'dxf_path': dxf_path
        }

    def _calculate_confidence(self, foot_mask, calibration_data):
        """Calcule le score de confiance"""
        confidence = 50
        
        if foot_mask is not None:
            confidence += 25
        
        if calibration_data and calibration_data.get('board_detected'):
            confidence += 25
            if calibration_data.get('pose_info') is not None:
                confidence += 10
        elif calibration_data and calibration_data.get('ratio_px_mm') is not None:
            confidence += 15
        
        return min(confidence, 100)

    # ============================================================
    # PIPELINES PUBLICS (SIDE & TOP)
    # ============================================================
    
    def process_side_view(self, image_path, debug=False, foot_side="right"):
        """
        PIPELINE SIDE VIEW : Sort uniquement la LONGUEUR.
        Utilise l'approche horizontale (X-axis) pour talon/orteil.
        """
        print(f"\n📱 PIPELINE SIDE: {os.path.basename(image_path)} (Pied {foot_side.upper()})")
        return self._run_pipeline(image_path, view_type='side', debug=debug, foot_side=foot_side)

    def process_top_view(self, image_path, debug=False, foot_side="right"):
        """
        PIPELINE TOP VIEW : Sort uniquement la LARGEUR.
        Utilise la largeur max (Left-Right).
        """
        print(f"\n📱 PIPELINE TOP: {os.path.basename(image_path)} (Pied {foot_side.upper()})")
        return self._run_pipeline(image_path, view_type='top', debug=debug, foot_side=foot_side)

    def _run_pipeline(self, image_path, view_type, debug=False, foot_side="right"):
        """Moteur commun pour l'exécution du pipeline"""
        # ... (début inchangé)
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f"Impossible de charger l'image: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        print(f"📸 Image: {w}x{h}px")

        if not self.initialized:
            return {'error': "SAM non initialisé"}

        # 1. Segmentation SAM
        print("🤖 Segmentation SAM...")
        masks = self.mask_generator.generate(image_rgb)
        if not masks:
            return {'error': "Aucun masque généré par SAM"}

        # 2. Calibration
        print("🎯 Calibration (ArUco / Carte)...")
        ratio_px_mm, calibration_data, aruco_markers = self._detect_aruco_l_board(image_rgb)
        
        # ... (Tentatives robustesse ArUco ajoutées précédemment)
        # Je réinclus le bloc calibration complet pour être sûr du contexte de l'edit si nécessaire, 
        # mais ici je vais cibler _measure_top_view_data call plus bas.
        
        # ... (skip calibration details for brevity in search match, assume context matches)

        calibration_method = "aruco" if ratio_px_mm else None
        
        if not ratio_px_mm:
            # Fallback Carte
            print("📏 Fallback: Détection carte de crédit...")
            foot_mask, card_mask, _ = self._identify_foot_and_card(masks, image_rgb)
            if card_mask is not None:
                 # ... (fallback code)
                 contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 if contours:
                    card_contour = max(contours, key=cv2.contourArea)
                    rect = cv2.minAreaRect(card_contour)
                    card_px = max(rect[1])
                    ratio_px_mm = card_px / CREDIT_CARD_WIDTH_MM
                    calibration_method = "credit_card"
                    calibration_data = {'ratio_px_mm': ratio_px_mm, 'board_detected': False}
                    print(f"✅ Carte détectée. Ratio: {ratio_px_mm:.3f} px/mm")

            if not ratio_px_mm:
                 return {'error': "Aucune référence (ArUco ou Carte) détectée"}

        # 3. Identification Pied
        foot_mask, _, _ = self._identify_foot_and_card(masks, image_rgb)
        if foot_mask is None:
            return {'error': "Pied non détecté"}

        # 4. Nettoyage spécifique
        if view_type == 'side':
            print("✂️ [Side] Affinement (suppression cheville)...")
            foot_mask = keep_foot_only(foot_mask, axis='y')
        
        # 5. Mesures Spécifiques
        if view_type == 'side':
            measurements = self._measure_side_view_data(foot_mask, ratio_px_mm, foot_side=foot_side)
            print(f"✅ SIDE Result: Longueur = {measurements.get('length_cm')} cm")
        else: # top
            measurements = self._measure_top_view_data(foot_mask, ratio_px_mm, foot_side=foot_side)
            print(f"✅ TOP Result: Largeur = {measurements.get('width_cm')} cm")
        
        if 'error' in measurements:
            return measurements

        # Métadonnées
        measurements.update({
            'image_path': image_path,
            'original_dimensions': f"{w}x{h}",
            'calibration_method': calibration_method,
            'confidence': self._calculate_confidence(foot_mask, calibration_data)
        })

        # 6. Debug
        if debug:
            self._save_debug_images(image_rgb, foot_mask, aruco_markers, measurements, calibration_data)

        return measurements

    # ============================================================
    # E) DEBUG
    # ============================================================
    
    def _save_debug_images(self, image, foot_mask, aruco_markers, measurements, calibration_data):
        """Sauvegarde les images de debug avec points dessinés"""
        debug_dir = f"output/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_dir, exist_ok=True)
        
        vis = image.copy()
        
        # Overlay masque pied (vert)
        if foot_mask is not None:
            foot_overlay = np.zeros_like(vis)
            foot_overlay[foot_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, foot_overlay, 0.3, 0)
        
        # Dessiner ArUco markers
        if aruco_markers is not None:
            for marker_id, corners in aruco_markers.items():
                corners_int = corners.astype(int)
                cv2.polylines(vis, [corners_int], True, (0, 255, 255), 3)
                center = np.mean(corners, axis=0).astype(int)
                cv2.putText(vis, f"ID:{marker_id}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Dessiner points talon/orteil
        if 'heel_point' in measurements and 'toe_point' in measurements:
            heel = tuple(map(int, measurements['heel_point']))
            toe = tuple(map(int, measurements['toe_point']))
            cv2.circle(vis, heel, 8, (255, 0, 0), -1)  # Rouge = talon
            cv2.circle(vis, toe, 8, (0, 0, 255), -1)   # Bleu = orteil
            cv2.line(vis, heel, toe, (255, 255, 0), 2)  # Jaune = longueur
        
        # Dessiner points largeur (pour top view)
        if 'left_point' in measurements and 'right_point' in measurements:
            left = tuple(measurements['left_point'])
            right = tuple(measurements['right_point'])
            cv2.circle(vis, left, 8, (255, 0, 0), -1)
            cv2.circle(vis, right, 8, (0, 0, 255), -1)
            cv2.line(vis, left, right, (0, 255, 255), 2)
        
        cv2.imwrite(f"{debug_dir}/calibration_debug.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Rapport texte
        with open(f"{debug_dir}/measurement_report.txt", 'w', encoding='utf-8') as f:
            f.write("FOOT MEASUREMENT REPORT\n")
            f.write("="*50 + "\n\n")
            if calibration_data:
                f.write(f"Calibration: {'ArUco' if calibration_data.get('board_detected') else 'Credit Card'}\n")
                f.write(f"Ratio (px/mm): {calibration_data.get('ratio_px_mm', 'N/A')}\n\n")
            f.write("MEASUREMENTS:\n")
            for key, value in measurements.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value}\n")
        
        print(f"📁 Debug saved to: {debug_dir}")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def quick_measure(image_path, view_type='side'):
    """Mesure rapide pour intégration mobile"""
    pipeline = MobileSAMPodiatryPipeline()
    
    if not pipeline.initialized:
        return {'error': 'Pipeline non initialisé - SAM requis'}
    
    if view_type == 'top':
        return pipeline.process_top_view(image_path, debug=False)
    else:
        return pipeline.process_side_view(image_path, debug=False)


def batch_process_folder(folder_path, output_csv=None, view_type='side'):
    """Traite tous les images d'un dossier"""
    import glob
    import pandas as pd
    
    pipeline = MobileSAMPodiatryPipeline()
    
    if not pipeline.initialized:
        print("❌ Pipeline non initialisé")
        return
    
    patterns = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
    
    if not image_files:
        print(f"❌ Aucune image trouvée dans {folder_path}")
        return
    
    print(f"📁 {len(image_files)} images à traiter ({view_type})")
    
    results = []
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] {os.path.basename(image_path)}")
        
        try:
            if view_type == 'top':
                measurements = pipeline.process_top_view(image_path, debug=True)
            else:
                measurements = pipeline.process_side_view(image_path, debug=True)
            
            if 'error' not in measurements:
                res = {'filename': os.path.basename(image_path)}
                res.update(measurements)
                results.append(res)
                
                if view_type == 'side':
                    print(f"✅ OK: Longueur {measurements.get('length_cm')}cm")
                else:
                    print(f"✅ OK: Largeur {measurements.get('width_cm')}cm")
            else:
                print(f"❌ Erreur: {measurements['error']}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    if results:
        df = pd.DataFrame(results)
        
        if output_csv:
            df.to_csv(output_csv, index=False)
            print(f"\n📊 Résultats sauvegardés: {output_csv}")

        
        print("\n📊 STATISTIQUES:")
        print(f"Images traitées: {len(results)}/{len(image_files)}")
        print(f"Longueur moyenne: {df['length_cm'].mean():.1f} cm")
        print(f"Largeur moyenne: {df['width_cm'].mean():.1f} cm")


def validate_setup():
    """Vérifie que tout est correctement installé"""
    print("🔍 Vérification de l'installation...\n")
    
    import sys
    print(f"✓ Python {sys.version.split()[0]}")
    
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'segment_anything': 'segment-anything',
        'ezdxf': 'ezdxf'
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

