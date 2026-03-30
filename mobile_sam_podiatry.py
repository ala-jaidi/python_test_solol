# mobile_sam_podiatry.py - Pipeline SAM simplifié pour application mobile podologue
# SIDE uniquement via process_foot_image avec _find_heel_and_toe (talon=max Y, orteil=min Y)

import os

# Optimisation threads CPU (A definir avant les autres imports)
_NUM_THREADS = int(os.environ.get("OMP_NUM_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(_NUM_THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(_NUM_THREADS))

import cv2
import numpy as np
import torch
torch.set_num_threads(_NUM_THREADS)
from datetime import datetime
from utils import keep_foot_only
from dxf_export import DXFExporter

# ============================================================
# A) CHARGEMENT / SAM
# ============================================================

# SAM imports avec gestion d'erreur
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("⚠️ SAM non disponible - installer avec: pip install segment-anything")

# ArUco L-shaped board configuration
ARUCO_L_BOARD_SIZE_MM = 100.0
ARUCO_L_BOARD_SEPARATION_MM = 20.0
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
                self.sam.eval()
                
                # Optimisation CPU : Quantification Dynamique (int8)
                if self.device == "cpu":
                    print("⚡ Application de la quantification dynamique (int8) sur l'Image Encoder...")
                    self.sam.image_encoder = torch.quantization.quantize_dynamic(
                        self.sam.image_encoder,
                        {torch.nn.Linear},
                        dtype=torch.qint8
                    )
                
                # 1. Générateur "Coarse" (Rapide, détection globale)
                self.mask_generator_coarse = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=12,  # Réduit pour vitesse (Passe 1)
                    pred_iou_thresh=0.86,
                    stability_score_thresh=0.90,
                    crop_n_layers=0,
                    min_mask_region_area=1000,
                    box_nms_thresh=0.7
                )

                # 2. Predictor "Fine" (Précis, piloté par points intelligents)
                self.predictor = SamPredictor(self.sam)
                
                self.initialized = True
                print(f"✅ Pipeline SAM unifié initialisé ({self.device}) - Mode Coarse-to-Fine (Smart Points)")
                
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
        
        if len(marker_positions) < 2:
            print("⚠️ ArUco L-board: At least 2 markers required")
            return None, None, None

        # Stratégie de paires pour la calibration
        # 0: Coin, 1: X-axis, 2: Y-axis
        # Distance (0-1) = Distance (0-2) = known_distance_mm
        # Distance (1-2) = known_distance_mm * sqrt(2)
        
        known_distance_mm = ARUCO_L_BOARD_SIZE_MM + ARUCO_L_BOARD_SEPARATION_MM
        distance_px = 0
        used_pair = ""

        if 0 in marker_positions and 1 in marker_positions:
            # Cas optimal : 0 et 1 (Axe principal)
            p0 = np.mean(marker_positions[0], axis=0)
            p1 = np.mean(marker_positions[1], axis=0)
            distance_px = np.linalg.norm(p1 - p0)
            used_pair = "0-1"
            
        elif 0 in marker_positions and 2 in marker_positions:
            # Fallback : 0 et 2 (Axe secondaire)
            p0 = np.mean(marker_positions[0], axis=0)
            p2 = np.mean(marker_positions[2], axis=0)
            distance_px = np.linalg.norm(p2 - p0)
            used_pair = "0-2"
            
        elif 1 in marker_positions and 2 in marker_positions:
            # Fallback : 1 et 2 (Diagonale)
            p1 = np.mean(marker_positions[1], axis=0)
            p2 = np.mean(marker_positions[2], axis=0)
            distance_px = np.linalg.norm(p2 - p1)
            # Ajustement de la distance réelle pour la diagonale
            known_distance_mm = known_distance_mm * np.sqrt(2)
            used_pair = "1-2"
        else:
            print("⚠️ ArUco L-board: No valid pair found (0-1, 0-2, or 1-2)")
            return None, None, None

        ratio_px_mm = distance_px / known_distance_mm
        
        calibration_data = {
            'ratio_px_mm': ratio_px_mm,
            'marker_positions': marker_positions,
            'distance_px': distance_px,
            'known_distance_mm': known_distance_mm,
            'board_detected': True
        }
        
        print(f"✅ ArUco L-board detected ({used_pair}): {distance_px:.1f}px = {known_distance_mm:.1f}mm")
        print(f"📏 Ratio: {ratio_px_mm:.3f} px/mm")
        
        return ratio_px_mm, calibration_data, marker_positions

    def _predict_fine_foot(self, foot_crop, foot_mask_small, scale, crop_offset):
        """
        Segmentation Fine via SamPredictor avec points intelligents.
        Reproduit la logique 'AutomaticMaskGenerator' mais ciblée et légère.
        """
        # 1. Image Encoder (lourd, mais fait une seule fois sur le crop)
        self.predictor.set_image(foot_crop)
        
        h_crop, w_crop = foot_crop.shape[:2]
        h_small, w_small = foot_mask_small.shape[:2]
        crop_x1, crop_y1 = crop_offset
        
        points = []
        labels = []
        
        # Grille 10x10 sur le crop (environ 100 points potentiels, filtrés par le masque coarse)
        # Cela assure une couverture "jusqu'aux bords" comme demandé.
        grid_steps = 10
        xs = np.linspace(0, w_crop - 1, grid_steps)
        ys = np.linspace(0, h_crop - 1, grid_steps)
        
        for x in xs:
            for y in ys:
                global_x = x + crop_x1
                global_y = y + crop_y1
                
                # Conversion coordonnées vers masque small
                sx = int(global_x * scale)
                sy = int(global_y * scale)
                
                if 0 <= sx < w_small and 0 <= sy < h_small:
                    # Si le point tombe dans le masque grossier -> Point Positif
                    if foot_mask_small[sy, sx] > 0:
                        points.append([x, y])
                        labels.append(1)
        
        if not points:
            # Fallback: centre du crop si échec alignement
            points.append([w_crop/2, h_crop/2])
            labels.append(1)

        points_np = np.array(points)
        labels_np = np.array(labels)
        
        # 3. Prédiction avec SamPredictor
        # multimask_output=True laisse SAM proposer 3 variantes (Part, Whole, etc.)
        masks, scores, logits = self.predictor.predict(
            point_coords=points_np,
            point_labels=labels_np,
            multimask_output=True
        )
        
        # On garde le masque avec le score de confiance le plus élevé
        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]
        
        # Nettoyage morphologique rapide
        mask_uint8 = (best_mask * 255).astype(np.uint8)
        return self._clean_mask(mask_uint8)

    # ============================================================
    # C) SEGMENTATION + NETTOYAGE
    # ============================================================
    
    def _identify_foot(self, masks, image):
        """Identifie le pied parmi les masques SAM (simplifié, ArUco only)"""
        h, w = image.shape[:2]
        image_area = h * w
        foot_candidates = []
        
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']
            
            area_ratio = area / image_area
            bbox_w, bbox_h = bbox[2], bbox[3]
            aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if min(bbox_w, bbox_h) > 0 else 0
            
            if 0.08 <= area_ratio <= 0.45 and 1.8 <= aspect_ratio <= 4.5:
                if self._is_foot_like(mask_data, h, w):
                    foot_score = self._score_foot_candidate(mask_data, h, w)
                    foot_candidates.append((mask, foot_score, mask_data))
        
        best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        
        # Fallback si pas de pied trouvé (critères relaxés)
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
        
        if best_foot:
            foot_mask = (best_foot[0] * 255).astype(np.uint8)
            return self._clean_mask(foot_mask)
        return None

    def _is_foot_like(self, mask_data, H, W):
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

        # 1. Segmentation SAM (Stratégie Coarse-to-Fine)
        print("🤖 Segmentation SAM (Mode Double Passe)...")
        
        # --- PASSE 1: COARSE (Image réduite) ---
        target_dim = 1024
        scale = target_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_small = cv2.resize(image_rgb, (new_w, new_h))
        
        with torch.inference_mode():
            masks_coarse = self.mask_generator_coarse.generate(image_small)
            
        if not masks_coarse:
             return {'error': "Aucun masque généré (Passe 1)"}
             
        # Identifier pied sur basse résolution
        foot_mask_small = self._identify_foot(masks_coarse, image_small)
        
        if foot_mask_small is None:
             return {'error': "Pied non détecté (Passe 1)"}

        # --- PASSE 2: FINE (Crop haute résolution) ---
        # Calculer BBox du pied sur l'image originale
        y_idx, x_idx = np.where(foot_mask_small > 0)
        if len(y_idx) == 0:
            return {'error': "Masque pied vide"}
            
        y_min, y_max = y_idx.min(), y_idx.max()
        x_min, x_max = x_idx.min(), x_idx.max()
        
        # Remise à l'échelle
        y_min_orig = int(y_min / scale)
        y_max_orig = int(y_max / scale)
        x_min_orig = int(x_min / scale)
        x_max_orig = int(x_max / scale)
        
        # Marge de sécurité (15%)
        margin_h = int((y_max_orig - y_min_orig) * 0.15)
        margin_w = int((x_max_orig - x_min_orig) * 0.15)
        
        crop_y1 = max(0, y_min_orig - margin_h)
        crop_y2 = min(h, y_max_orig + margin_h)
        crop_x1 = max(0, x_min_orig - margin_w)
        crop_x2 = min(w, x_max_orig + margin_w)
        
        foot_crop = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        print(f"🔍 Fine Pass Crop: {foot_crop.shape[1]}x{foot_crop.shape[0]} px")
        
        foot_mask = None
        if foot_crop.size > 0:
            try:
                # Utilisation de SamPredictor avec points intelligents
                foot_mask_fine = self._predict_fine_foot(
                    foot_crop, 
                    foot_mask_small, 
                    scale, 
                    (crop_x1, crop_y1)
                )
                
                if foot_mask_fine is not None:
                    # Reconstruire le masque complet
                    foot_mask = np.zeros((h, w), dtype=np.uint8)
                    foot_mask[crop_y1:crop_y2, crop_x1:crop_x2] = foot_mask_fine
                else:
                    print("⚠️ Echec Fine Pass: aucun masque retourné")
            except Exception as e:
                print(f"⚠️ Exception Fine Pass: {e}")

        # Fallback: Upscale du masque grossier si le fin a échoué
        if foot_mask is None:
             print("⚠️ Utilisation du masque grossier (Upscaled)")
             foot_mask = cv2.resize(foot_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # 2. Calibration
        print("🎯 Calibration (ArUco / Carte)...")
        ratio_px_mm, calibration_data, aruco_markers = self._detect_aruco_l_board(image_rgb)

        if not ratio_px_mm:
            return {'error': "ArUco non détecté - calibration impossible"}
        
        calibration_method = "aruco"

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
            debug_path = self._save_debug_images(image_rgb, foot_mask, aruco_markers, measurements, calibration_data)
            measurements['debug_image_path'] = debug_path

        return measurements

    # ============================================================
    # E) DEBUG
    # ============================================================
    
    def _save_debug_images(self, image, foot_mask, aruco_markers, measurements, calibration_data):
        """Sauvegarde les images de debug avec points dessinés"""
        import uuid
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        unique_id = str(uuid.uuid4())[:8]
        debug_dir = f"output/debug_{timestamp}_{unique_id}"
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
        return f"{debug_dir}/calibration_debug.jpg"


