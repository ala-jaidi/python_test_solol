# mobile_sam_podiatry.py - Pipeline unifié SAM pour application mobile podologue
# Segmentation pied + carte de crédit + correction perspective + mesures

import cv2
import numpy as np
import os
import torch
from datetime import datetime
from scipy.spatial import distance

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
ARUCO_L_BOARD_SIZE_MM = 100.0  # Size of each ArUco marker in mm
ARUCO_L_BOARD_SEPARATION_MM = 20.0  # Separation between markers in mm
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # ArUco dictionary

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
                
                # Stricter mask generator to reduce large background masks
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=self.sam,
                    points_per_side=20,
                    pred_iou_thresh=0.90,
                    stability_score_thresh=0.92,
                    crop_n_layers=0,              # no crops
                    min_mask_region_area=1200,    # avoid ultra-thin sheets
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

        # 3. Try ArUco L-board detection first (preferred method)
        print("🎯 Tentative détection ArUco L-board...")
        ratio_px_mm, calibration_data, aruco_markers = self._detect_aruco_l_board(image_rgb)
        
        calibration_method = None
        if ratio_px_mm is not None:
            calibration_method = "aruco"
            print("✅ Calibration ArUco 3D réussie")
        else:
            # 4. Fallback to credit card detection
            print("📏 Fallback: Détection carte de crédit...")
            foot_mask, card_mask, _ = self._identify_foot_and_card(masks, image_rgb)
            
            if card_mask is None:
                return {'error': "Aucune référence détectée (ArUco ou carte)"}
            
            contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return {'error': "Contour carte non trouvé"}
            card_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(card_contour)
            card_px = max(rect[1])  # largeur ou hauteur en pixels
            ratio_px_mm = card_px / CREDIT_CARD_WIDTH_MM
            calibration_method = "credit_card"
            calibration_data = {
                'ratio_px_mm': ratio_px_mm,
                'board_detected': False,
                'pose_info': None
            }
            print(f"✅ Largeur carte: {card_px:.1f}px = {CREDIT_CARD_WIDTH_MM}mm → ratio: {ratio_px_mm:.3f} px/mm")
        
        # Detect foot regardless of calibration method
        foot_mask, _, _ = self._identify_foot_and_card(masks, image_rgb)
        if foot_mask is None:
            return {'error': "Pied non détecté"}

        # 5. Mesurer le pied avec calibration 3D si disponible
        if calibration_method == "aruco":
            # Use 3D-aware measurements
            foot_contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if foot_contours:
                foot_contour = max(foot_contours, key=cv2.contourArea)
                measurements = self._calculate_3d_measurements(foot_contour, calibration_data, 'general')
                # Add additional standard measurements
                std_measurements = self._measure_foot(image_rgb, foot_mask, ratio_px_mm)
                measurements.update({
                    'width_cm': std_measurements['width_cm'],
                    'area_cm2': std_measurements['area_cm2'],
                    'perimeter_cm': std_measurements['perimeter_cm'],
                    'length_width_ratio': std_measurements['length_width_ratio']
                })
            else:
                return {'error': "Contour du pied non trouvé"}
        else:
            # Use standard 2D measurements
            measurements = self._measure_foot(image_rgb, foot_mask, ratio_px_mm)

        # 6. Ajouter métadonnées
        measurements.update({
            'image_path': image_path,
            'original_dimensions': f"{w}x{h}",
            'perspective_corrected': False,
            'calibration_method': calibration_method,
            'aruco_detected': calibration_method == "aruco",
            'card_detected': calibration_method == "credit_card",
            'processing_timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_confidence_new(foot_mask, calibration_data)
        })

        # 7. Debug si demandé
        if debug:
            reference_mask = None
            if calibration_method == "credit_card":
                # Find card mask for debug
                _, reference_mask, _ = self._identify_foot_and_card(masks, image_rgb)
            
            self._save_debug_images_new(
                image_rgb, foot_mask, reference_mask, aruco_markers,
                measurements, calibration_data
            )

        print(f"✅ Mesures terminées: {measurements['length_cm']:.1f}cm x {measurements['width_cm']:.1f}cm")
        return measurements
    
    
    def _identify_foot_and_card(self, masks, image):
        """Identifie le pied et la carte de crédit parmi les masques SAM"""
        h, w = image.shape[:2]
        image_area = h * w
        
        foot_candidates = []
        card_candidates = []
        
        # 1) Collecter les meilleurs candidats carte d'abord
        for mask_data in masks:
            area_ratio = mask_data['area'] / image_area
            if 0.005 <= area_ratio <= 0.15:  # Carte plus petite que le pied
                card_score = self._score_card_candidate(mask_data, h, w)
                if card_score > 0:
                    card_candidates.append((mask_data['segmentation'], card_score, mask_data))
        
        best_card = max(card_candidates, key=lambda x: x[1]) if card_candidates else None
        card_mask = None
        card_corners = None
        if best_card:
            card_mask = (best_card[0] * 255).astype(np.uint8)
            card_corners = self._find_card_corners(card_mask)
        
        # 2) Ensuite, filtrer/empiler les candidats pied avec _is_foot_like
        for mask_data in masks:
            mask = mask_data['segmentation']
            area = mask_data['area']
            bbox = mask_data['bbox']  # x, y, w, h
            
            area_ratio = area / image_area
            bbox_w, bbox_h = bbox[2], bbox[3]
            aspect_ratio = max(bbox_w, bbox_h) / min(bbox_w, bbox_h) if min(bbox_w, bbox_h) > 0 else 0
            
            if 0.08 <= area_ratio <= 0.45 and 1.8 <= aspect_ratio <= 4.5:
                # Appliquer les filtres rapides
                if self._is_foot_like(mask_data, h, w, card_mask=card_mask):
                    foot_score = self._score_foot_candidate(mask_data, h, w)
                    foot_candidates.append((mask, foot_score, mask_data))
        
        best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        
        # 2b) Fallback: si aucun pied trouvé, relâcher les seuils
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
                # Seuils plus permissifs
                if 0.05 <= area_ratio <= 0.60 and 1.2 <= aspect_ratio <= 5.0:
                    # On ne force pas _is_foot_like ici pour éviter faux négatifs
                    foot_score = self._score_foot_candidate(mask_data, h, w)
                    foot_candidates.append((mask, foot_score * 0.9, mask_data))  # légère pénalité
            best_foot = max(foot_candidates, key=lambda x: x[1]) if foot_candidates else None
        
        foot_mask = None
        if best_foot:
            foot_mask = (best_foot[0] * 255).astype(np.uint8)
            foot_mask = self._clean_mask(foot_mask)
        
        return foot_mask, card_mask, card_corners

    def _is_foot_like(self, mask_data, H, W, card_mask=None):
        m = (mask_data['segmentation'] * 255).astype(np.uint8)
        x, y, w, h = mask_data['bbox']
        
        # 1) rejet des bandes trop plates (sol)
        if h / H < 0.22:  # au moins 22% de la hauteur image
            return False
        
        # 2) ne pas toucher >= 2 bords de l'image (typique du sol)
        touches = int(y <= 2) + int(y + h >= H - 3) + int(x <= 2) + int(x + w >= W - 3)
        if touches >= 2:
            return False
        
        # 3) filtre PCA : bande ultra-plate -> rejet
        ys, xs = np.where(m > 0)
        if xs.size < 20:
            return False
        pts = np.c_[xs, ys].astype(np.float32)
        cov = np.cov(pts.T)
        eigvals, _ = np.linalg.eigh(cov)
        flat = eigvals.min() / eigvals.max()
        if flat < 0.12:  # trop "ruban"
            return False
        
        # 4) ne pas recouvrir la carte (si on l’a)
        if card_mask is not None:
            overlap = np.logical_and(m > 0, card_mask > 0).sum() / max((m > 0).sum(), 1)
            if overlap > 0.05:
                return False
        return True
    
    def _score_foot_candidate(self, mask_data, h, w):
        """Enhanced foot candidate scoring with improved robustness"""
        score = 0
        area_ratio = mask_data['area'] / (h * w)
        bbox = mask_data['bbox']
        aspect_ratio = max(bbox[2], bbox[3]) / min(bbox[2], bbox[3])
        
        # Enhanced area scoring with adaptive thresholds
        if 0.12 <= area_ratio <= 0.35:
            score += 35  # Ideal foot size
        elif 0.08 <= area_ratio <= 0.50:
            score += 20  # Acceptable range
        elif 0.05 <= area_ratio <= 0.60:
            score += 10  # Extended range for different poses
        
        # Improved aspect ratio scoring
        if 1.5 <= aspect_ratio <= 3.5:
            # More flexible for different foot orientations
            optimal_score = 30 * (1 - abs(aspect_ratio - 2.5) / 2.0)
            score += max(optimal_score, 15)
        elif 1.2 <= aspect_ratio <= 4.5:
            score += 10
        
        # Position scoring with better tolerance
        center_x = bbox[0] + bbox[2]/2
        center_y = bbox[1] + bbox[3]/2
        
        # Prefer center to lower-center positioning
        if 0.3 * h <= center_y <= 0.8 * h:
            score += 20
        elif 0.2 * h <= center_y <= 0.9 * h:
            score += 10
        
        # Horizontal centering (less strict)
        if 0.2 * w <= center_x <= 0.8 * w:
            score += 15
        
        # Enhanced stability scoring
        stability = mask_data.get('stability_score', 0)
        score += stability * 35
        
        # Shape complexity analysis
        mask = mask_data['segmentation']
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            contour = max(contours, key=cv2.contourArea)
            
            # Convexity analysis (feet should be moderately convex)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour)
            
            if hull_area > 0:
                convexity = contour_area / hull_area
                if 0.7 <= convexity <= 0.95:  # Feet have some concavity
                    score += 20
                elif 0.6 <= convexity <= 0.98:
                    score += 10
            
            # Perimeter to area ratio (shape complexity)
            perimeter = cv2.arcLength(contour, True)
            if contour_area > 0:
                complexity = perimeter * perimeter / contour_area
                if 15 <= complexity <= 40:  # Typical foot complexity
                    score += 15
        
        # Predicted IoU bonus
        predicted_iou = mask_data.get('predicted_iou', 0)
        score += predicted_iou * 20
        
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
            corners = corners.astype(np.int32)
        
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
    
    def _detect_aruco_l_board(self, image):
        """
        Detect ArUco L-shaped board and calculate 3D-aware calibration
        Returns ratio_px_mm and pose information for 3D measurements
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is None or len(ids) < 2:
            return None, None, None
        
        # For L-shaped board, we expect at least 2 markers with specific IDs
        # Marker 0: Origin (corner of L)
        # Marker 1: Along X-axis
        # Marker 2: Along Y-axis (optional for validation)
        
        marker_positions = {}
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2]:  # Only process our L-board markers
                marker_positions[marker_id] = corners[i][0]
        
        if 0 not in marker_positions or 1 not in marker_positions:
            print("⚠️ ArUco L-board: Markers 0 and 1 required")
            return None, None, None
        
        # Calculate distances and scaling
        marker0_center = np.mean(marker_positions[0], axis=0)
        marker1_center = np.mean(marker_positions[1], axis=0)
        
        # Distance between markers in pixels
        distance_px = np.linalg.norm(marker1_center - marker0_center)
        
        # Known distance in mm (center to center)
        known_distance_mm = ARUCO_L_BOARD_SIZE_MM + ARUCO_L_BOARD_SEPARATION_MM
        
        # Calculate pixel to mm ratio
        ratio_px_mm = distance_px / known_distance_mm
        
        # Calculate pose information for 3D awareness
        # Create 3D points for L-board (assuming board is on ground plane, Z=0)
        object_points = np.array([
            [0, 0, 0],  # Marker 0 origin
            [known_distance_mm, 0, 0],  # Marker 1 along X
            [0, known_distance_mm, 0]   # Marker 2 along Y (if present)
        ], dtype=np.float32)
        
        # Image points (marker centers)
        image_points = np.array([
            marker0_center,
            marker1_center
        ], dtype=np.float32)
        
        if 2 in marker_positions:
            marker2_center = np.mean(marker_positions[2], axis=0)
            image_points = np.vstack([image_points, marker2_center])
            object_points_used = object_points
        else:
            object_points_used = object_points[:2]
        
        # Camera parameters (approximate - for better results, camera should be calibrated)
        image_size = gray.shape[::-1]
        focal_length = max(image_size)  # Approximate focal length
        camera_matrix = np.array([
            [focal_length, 0, image_size[0]/2],
            [0, focal_length, image_size[1]/2],
            [0, 0, 1]
        ], dtype=np.float32)
        
        dist_coeffs = np.zeros((4, 1))  # Assume no distortion
        
        # Solve PnP to get pose
        pose_info = None
        if len(image_points) >= 3:
            try:
                success, rvec, tvec = cv2.solvePnP(
                    object_points_used, image_points, camera_matrix, dist_coeffs
                )
                if success:
                    pose_info = {
                        'rvec': rvec,
                        'tvec': tvec,
                        'camera_matrix': camera_matrix
                    }
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
        print(f"📏 3D-aware ratio: {ratio_px_mm:.3f} px/mm")
        
        return ratio_px_mm, calibration_data, marker_positions

    def _calculate_3d_measurements(self, foot_contour, calibration_data, view_type='side'):
        """
        Calculate measurements using 3D-aware calibration data
        """
        ratio_px_mm = calibration_data['ratio_px_mm']
        pose_info = calibration_data.get('pose_info')
        
        if view_type == 'side' and pose_info is not None:
            # For side view, we can use pose information to correct for viewing angle
            return self._calculate_side_view_3d(foot_contour, calibration_data)
        else:
            # Fallback to 2D measurements
            return self._calculate_2d_measurements(foot_contour, ratio_px_mm)
    
    def _calculate_side_view_3d(self, foot_contour, calibration_data):
        """
        Calculate side view measurements with 3D pose correction
        """
        ratio_px_mm = calibration_data['ratio_px_mm']
        pose_info = calibration_data['pose_info']
        
        # Get basic 2D measurements first
        pts = foot_contour[:, 0, :]
        
        # Find key points
        heel = pts[pts[:, 0].argmin()]  # Leftmost point
        toe = pts[pts[:, 0].argmax()]   # Rightmost point
        
        # Ground line (bottom of foot)
        bottom_points = pts[pts[:, 1] >= pts[:, 1].max() - 10]
        ground_y = int(np.median(bottom_points[:, 1]))
        
        # Arch point (highest in middle region)
        mid_x_start = heel[0] + (toe[0] - heel[0]) * 0.2
        mid_x_end = heel[0] + (toe[0] - heel[0]) * 0.8
        mid_pts = pts[(pts[:, 0] >= mid_x_start) & (pts[:, 0] <= mid_x_end)]
        
        if len(mid_pts) > 0:
            arch = mid_pts[mid_pts[:, 1].argmin()]
        else:
            arch = pts[pts[:, 1].argmin()]
        
        # Calculate measurements
        length_px = np.linalg.norm(toe - heel)
        arch_height_px = ground_y - arch[1]
        
        # Apply 3D correction if pose is available
        if pose_info is not None:
            # Extract rotation matrix from pose
            rvec = pose_info['rvec']
            R, _ = cv2.Rodrigues(rvec)
            
            # Estimate viewing angle correction factor
            # This is a simplified approach - more sophisticated correction can be added
            angle_factor = abs(R[0, 2])  # X-axis rotation component
            correction_factor = 1.0 / (1.0 - 0.3 * angle_factor)  # Empirical correction
            
            length_px *= correction_factor
            print(f"📐 3D correction applied: factor = {correction_factor:.3f}")
        
        # Convert to cm
        length_cm = (length_px / ratio_px_mm) / 10
        arch_height_cm = (arch_height_px / ratio_px_mm) / 10
        
        # Calculate arch angle
        arch_angle_rad = np.arctan2(arch_height_px, arch[0] - heel[0])
        arch_angle_deg = np.degrees(arch_angle_rad)
        
        return {
            'length_cm': round(length_cm, 2),
            'arch_height_cm': round(arch_height_cm, 2),
            'arch_angle_deg': round(arch_angle_deg, 2),
            'heel_point': heel.tolist(),
            'arch_point': arch.tolist(),
            'toe_point': toe.tolist(),
            'ground_y': int(ground_y),
            '3d_corrected': pose_info is not None
        }
    
    def _calculate_2d_measurements(self, foot_contour, ratio_px_mm):
        """
        Fallback 2D measurements when 3D pose is not available
        """
        pts = foot_contour[:, 0, :]
        
        # Basic measurements
        heel = pts[pts[:, 0].argmin()]
        toe = pts[pts[:, 0].argmax()]
        length_px = np.linalg.norm(toe - heel)
        length_cm = (length_px / ratio_px_mm) / 10
        
        return {
            'length_cm': round(length_cm, 2),
            'heel_point': heel.tolist(),
            'toe_point': toe.tolist(),
            '3d_corrected': False
        }
    
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
        
        return measurements
    
    def _find_real_width(self, foot_mask, contour):
        """Trouve la largeur réelle maximale du pied"""
        h, w = foot_mask.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        max_width = 0
        for y in range(h):
            row = mask[y, :]
            if np.any(row):
                indices = np.where(row)[0]
                width = indices[-1] - indices[0]
                max_width = max(max_width, width)

        return max_width

    def _find_max_width_points(self, foot_mask, contour):
        """Retourne la largeur maximale et les points gauche/droit correspondants"""
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
    
    def _find_heel_and_toe(self, contour):
        """Trouve les points talon et orteil"""
        # Point le plus bas = talon
        heel_idx = contour[:, :, 1].argmax()
        heel_point = contour[heel_idx, 0]
        
        # Point le plus haut = orteil
        toe_idx = contour[:, :, 1].argmin()
        toe_point = contour[toe_idx, 0]
        
        return heel_point, toe_point
    
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
    
    def _calculate_confidence_new(self, foot_mask, calibration_data):
        """Calculate confidence score for new calibration system"""
        confidence = 50  # Base confidence
        
        if foot_mask is not None:
            confidence += 25
        
        if calibration_data.get('board_detected'):
            confidence += 25  # ArUco is more reliable than credit card
            if calibration_data.get('pose_info') is not None:
                confidence += 10  # 3D pose adds extra confidence
        elif calibration_data.get('ratio_px_mm') is not None:
            confidence += 15  # Credit card fallback
        
        return min(confidence, 100)

    # -----------------------------------------------------.

    def _measure_side_view(self, image, foot_mask, ratio_px_mm):
        """Enhanced side view analysis with improved ground line and arch detection"""
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'error': 'Contour du pied non trouvé'}

        contour = max(contours, key=cv2.contourArea)
        pts = contour[:, 0, :]

        # Enhanced ground line detection
        ground_y = self._detect_robust_ground_line(pts, image.shape)

        # Improved heel and toe detection
        heel, forefoot = self._detect_heel_toe_robust(pts, ground_y)

        # Enhanced arch detection
        arch, instep_height = self._detect_arch_robust(pts, heel, forefoot, ground_y)

        # Measurements
        length_px = np.linalg.norm(forefoot - heel)
        arch_height_px = instep_height
        
        # Improved arch angle calculation
        arch_angle_rad = self._calculate_arch_angle(heel, arch, forefoot, ground_y)

        return {
            'length_cm': round((length_px / ratio_px_mm) / 10, 2),
            'arch_height_cm': round((arch_height_px / ratio_px_mm) / 10, 2),
            'instep_height_cm': round((instep_height / ratio_px_mm) / 10, 2),
            'arch_angle_deg': round(np.degrees(arch_angle_rad), 2),
            'heel_point': heel.tolist(),
            'arch_point': arch.tolist(),
            'forefoot_point': forefoot.tolist(),
            'ground_y': int(ground_y),
            'ratio_px_mm': round(ratio_px_mm, 3)
        }

    def _detect_robust_ground_line(self, pts, image_shape):
        """Robust ground line detection using multiple methods"""
        h, w = image_shape[:2]
        
        # Method 1: Bottom percentile approach
        bottom_10_percent = pts[pts[:, 1] >= np.percentile(pts[:, 1], 90)]
        ground_y_method1 = np.median(bottom_10_percent[:, 1])
        
        # Method 2: Histogram-based approach
        y_coords = pts[:, 1]
        hist, bins = np.histogram(y_coords, bins=max(10, int(h/20)))
        # Find the peak in the bottom third of the image
        bottom_third_start = int(len(bins) * 0.67)
        bottom_region_hist = hist[bottom_third_start:]
        if len(bottom_region_hist) > 0:
            peak_idx = np.argmax(bottom_region_hist) + bottom_third_start
            ground_y_method2 = bins[peak_idx]
        else:
            ground_y_method2 = ground_y_method1
        
        # Method 3: Linear regression on bottom points
        bottom_points = pts[pts[:, 1] >= np.percentile(pts[:, 1], 85)]
        if len(bottom_points) >= 3:
            try:
                from sklearn.linear_model import RANSACRegressor
                X = bottom_points[:, 0].reshape(-1, 1)
                y = bottom_points[:, 1]
                ransac = RANSACRegressor(residual_threshold=5.0)
                ransac.fit(X, y)
                # Get ground line at foot center
                foot_center_x = np.mean(pts[:, 0])
                ground_y_method3 = ransac.predict([[foot_center_x]])[0]
            except:
                ground_y_method3 = ground_y_method1
        else:
            ground_y_method3 = ground_y_method1
        
        # Combine methods with weights
        weights = [0.4, 0.3, 0.3]
        ground_y = (weights[0] * ground_y_method1 + 
                   weights[1] * ground_y_method2 + 
                   weights[2] * ground_y_method3)
        
        return int(ground_y)

    def _detect_heel_toe_robust(self, pts, ground_y):
        """Improved heel and toe detection"""
        # Find points near the ground line
        ground_tolerance = 15
        ground_points = pts[np.abs(pts[:, 1] - ground_y) <= ground_tolerance]
        
        if len(ground_points) < 2:
            # Fallback to extreme points
            heel = pts[pts[:, 0].argmin()]
            toe = pts[pts[:, 0].argmax()]
        else:
            # Use ground-level extreme points
            heel = ground_points[ground_points[:, 0].argmin()]
            toe = ground_points[ground_points[:, 0].argmax()]
        
        return heel, toe

    def _detect_arch_robust(self, pts, heel, toe, ground_y):
        """Enhanced arch detection with multiple validation methods"""
        # Define arch region (middle 40% of foot length)
        foot_length = toe[0] - heel[0]
        arch_start = heel[0] + foot_length * 0.3
        arch_end = heel[0] + foot_length * 0.7
        
        arch_region = pts[(pts[:, 0] >= arch_start) & (pts[:, 0] <= arch_end)]
        
        if len(arch_region) == 0:
            # Fallback to simple highest point
            arch = pts[pts[:, 1].argmin()]
            instep_height = ground_y - arch[1]
            return arch, instep_height
        
        # Method 1: Highest point in arch region
        arch_candidate1 = arch_region[arch_region[:, 1].argmin()]
        
        # Method 2: Smooth curve fitting
        try:
            # Sort arch region points by x-coordinate
            sorted_arch = arch_region[np.argsort(arch_region[:, 0])]
            
            # Fit polynomial to arch curve
            if len(sorted_arch) >= 5:
                z = np.polyfit(sorted_arch[:, 0], sorted_arch[:, 1], 2)
                p = np.poly1d(z)
                
                # Find minimum of parabola (highest point of arch)
                arch_x = -z[1] / (2 * z[0]) if z[0] != 0 else np.mean(sorted_arch[:, 0])
                arch_y = p(arch_x)
                arch_candidate2 = np.array([arch_x, arch_y])
            else:
                arch_candidate2 = arch_candidate1
        except:
            arch_candidate2 = arch_candidate1
        
        # Choose the higher (lower y-value) arch point
        if arch_candidate2[1] < arch_candidate1[1]:
            arch = arch_candidate2
        else:
            arch = arch_candidate1
        
        # Calculate instep height
        instep_height = ground_y - arch[1]
        
        # Validate arch height (should be reasonable)
        max_reasonable_height = abs(toe[0] - heel[0]) * 0.3  # Max 30% of foot length
        if instep_height > max_reasonable_height:
            # Use more conservative estimate
            conservative_points = arch_region[arch_region[:, 1] >= ground_y - max_reasonable_height]
            if len(conservative_points) > 0:
                arch = conservative_points[conservative_points[:, 1].argmin()]
                instep_height = ground_y - arch[1]
        
        return arch, max(instep_height, 0)

    def _calculate_arch_angle(self, heel, arch, toe, ground_y):
        """Calculate arch angle with improved accuracy"""
        # Method 1: Angle from heel to arch
        heel_to_arch = np.arctan2(arch[1] - ground_y, arch[0] - heel[0])
        
        # Method 2: Angle of arch relative to foot length
        foot_vector = toe - heel
        arch_vector = arch - heel
        
        # Project arch vector onto foot vector
        projection_length = np.dot(arch_vector, foot_vector) / np.linalg.norm(foot_vector)
        arch_height = ground_y - arch[1]
        
        # Calculate angle from projection
        if projection_length > 0:
            angle_from_projection = np.arctan2(arch_height, projection_length)
        else:
            angle_from_projection = heel_to_arch
        
        # Average the two methods
        final_angle = (heel_to_arch + angle_from_projection) / 2
        
        return final_angle

    def _calculate_toe_angle(self, contour, ratio_px_mm):
        """Enhanced toe angle calculation for foot progression angle"""
        pts = contour[:, 0, :]
        
        # Find foot axis (heel to toe direction)
        heel = pts[pts[:, 0].argmin()]
        toe = pts[pts[:, 0].argmax()]
        
        # Define toe region (front 20% of foot)
        foot_length = toe[0] - heel[0]
        toe_region_start = toe[0] - foot_length * 0.2
        toe_region = pts[pts[:, 0] >= toe_region_start]
        
        if len(toe_region) < 3:
            # Fallback: use simple heel-toe angle
            foot_axis_angle = np.degrees(np.arctan2(toe[1] - heel[1], toe[0] - heel[0]))
            return foot_axis_angle, 0  # No opening measurement
        
        # Method 1: Fit line to toe region
        try:
            from sklearn.linear_model import RANSACRegressor
            X = toe_region[:, 0].reshape(-1, 1)
            y = toe_region[:, 1]
            
            if len(X) >= 3:
                ransac = RANSACRegressor(residual_threshold=3.0)
                ransac.fit(X, y)
                toe_line_slope = ransac.estimator_.coef_[0]
                toe_line_angle = np.degrees(np.arctan(toe_line_slope))
            else:
                toe_line_angle = 0
        except:
            toe_line_angle = 0
        
        # Method 2: Principal component analysis of toe region
        try:
            toe_centered = toe_region - np.mean(toe_region, axis=0)
            cov_matrix = np.cov(toe_centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
            
            # Primary direction (largest eigenvalue)
            primary_direction = eigenvectors[:, np.argmax(eigenvalues)]
            pca_angle = np.degrees(np.arctan2(primary_direction[1], primary_direction[0]))
        except:
            pca_angle = toe_line_angle
        
        # Combine methods
        toe_angle = (toe_line_angle + pca_angle) / 2
        
        # Calculate foot progression angle (relative to image horizontal)
        foot_axis_angle = np.degrees(np.arctan2(toe[1] - heel[1], toe[0] - heel[0]))
        progression_angle = toe_angle - foot_axis_angle
        
        # Normalize angle to [-180, 180]
        while progression_angle > 180:
            progression_angle -= 360
        while progression_angle < -180:
            progression_angle += 360
        
        # Calculate toe width
        toe_width_px = np.max(toe_region[:, 0]) - np.min(toe_region[:, 0])
        toe_width_cm = (toe_width_px / ratio_px_mm) / 10
        
        return abs(progression_angle), toe_width_cm

    def _measure_top_view(self, image, foot_mask, ratio_px_mm):
        """Enhanced top view analysis with toe angle measurement"""
        contours, _ = cv2.findContours(foot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {'error': 'Contour du pied non détecté'}

        contour = max(contours, key=cv2.contourArea)
        width_px, left_pt, right_pt = self._find_max_width_points(foot_mask, contour)
        width_cm = (width_px / ratio_px_mm) / 10
        
        # Calculate toe angle (foot progression angle)
        toe_angle_deg, toe_width_cm = self._calculate_toe_angle(contour, ratio_px_mm)

        return {
            'width_cm': round(width_cm, 2),
            'toe_angle_deg': round(toe_angle_deg, 2),
            'toe_width_cm': round(toe_width_cm, 2),
            'left_point': list(left_pt),
            'right_point': list(right_pt),
            'ratio_px_mm': round(ratio_px_mm, 3)
        }

    def process_top_view_image(self, image_path, debug=False):
        """Traite une vue du dessus/empreinte"""
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f"Impossible de charger l'image: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.initialized:
            return {'error': 'SAM non initialisé'}

        masks = self.mask_generator.generate(image_rgb)
        if not masks:
            return {'error': 'Aucun masque généré par SAM'}

        foot_mask, card_mask, _ = self._identify_foot_and_card(masks, image_rgb)
        if foot_mask is None:
            return {'error': 'Pied non détecté'}
        if card_mask is None:
            return {'error': 'Carte non détectée'}

        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(card_contour)
        card_px = max(rect[1])
        ratio_px_mm = card_px / CREDIT_CARD_WIDTH_MM

        measures = self._measure_top_view(image_rgb, foot_mask, ratio_px_mm)
        measures.update({
            'image_path': image_path,
            'confidence': self._calculate_confidence(foot_mask, card_mask)
        })

        if debug:
            self._save_top_view_debug(image_rgb, foot_mask, card_mask, measures)

        return measures

    def process_side_view_image(self, image_path, debug=False):
        """Traite la vue de profil du pied"""
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f"Impossible de charger l'image: {image_path}"}

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not self.initialized:
            return {'error': 'SAM non initialisé'}

        masks = self.mask_generator.generate(image_rgb)
        if not masks:
            return {'error': 'Aucun masque généré par SAM'}

        foot_mask, card_mask, _ = self._identify_foot_and_card(masks, image_rgb)
        if foot_mask is None:
            return {'error': 'Pied non détecté'}
        if card_mask is None:
            return {'error': 'Carte non détectée'}

        contours, _ = cv2.findContours(card_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        card_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(card_contour)
        card_px = max(rect[1])
        ratio_px_mm = card_px / CREDIT_CARD_WIDTH_MM

        measures = self._measure_side_view(image_rgb, foot_mask, ratio_px_mm)
        measures.update({
            'image_path': image_path,
            'confidence': self._calculate_confidence(foot_mask, card_mask)
        })

        if debug:
            self._save_side_view_debug(image_rgb, foot_mask, card_mask, measures)

        return measures

    def process_hybrid_views(self, top_image_path, side_image_path, debug=False, foot_side='right'):
        """Combine top and side views to produce complete foot measurements in client's JSON format"""
        top = self.process_top_view_image(top_image_path, debug)
        if 'error' in top:
            return {'error': f"Top view: {top['error']}"}

        side = self.process_side_view_image(side_image_path, debug)
        if 'error' in side:
            return {'error': f"Side view: {side['error']}"}

        # Create the exact format requested by client
        result = {
            'length_cm': side['length_cm'],
            'width_cm': top['width_cm'],
            'instep_height_cm': side.get('instep_height_cm', side.get('arch_height_cm', 0)),
            'arch_angle_deg': side['arch_angle_deg'],
            'toe_angle_deg': top['toe_angle_deg'],
            'confidence': round((side.get('confidence', 50) + top.get('confidence', 50)) / 2, 1),
            'foot_side': foot_side,
            'calibration_method': {
                'top_view': 'aruco' if side.get('aruco_detected', False) else 'credit_card',
                'side_view': 'aruco' if side.get('aruco_detected', False) else 'credit_card'
            },
            'processing_timestamp': datetime.now().isoformat()
        }

        return result

    def _save_top_view_debug(self, image, foot_mask, card_mask, measures):
        """Enregistre le debug pour la vue du dessus"""
        debug_dir = f"output/debug_top_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_dir, exist_ok=True)
        vis = image.copy()

        if foot_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[foot_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        if card_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[card_mask > 0] = [255, 0, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        left = tuple(measures['left_point'])
        right = tuple(measures['right_point'])
        cv2.line(vis, left, right, (0, 255, 255), 2)
        cv2.circle(vis, left, 8, (255, 0, 0), -1)
        cv2.circle(vis, right, 8, (0, 0, 255), -1)

        cv2.imwrite(f"{debug_dir}/top_view.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def _save_side_view_debug(self, image, foot_mask, card_mask, measures):
        """Enregistre le debug pour la vue de profil"""
        debug_dir = f"output/debug_side_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_dir, exist_ok=True)
        vis = image.copy()

        if foot_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[foot_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        if card_mask is not None:
            overlay = np.zeros_like(vis)
            overlay[card_mask > 0] = [255, 0, 0]
            vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)

        heel = tuple(measures['heel_point'])
        arch = tuple(measures['arch_point'])
        fore = tuple(measures['forefoot_point'])
        ground_y = int(measures['ground_y'])

        cv2.line(vis, (0, ground_y), (vis.shape[1], ground_y), (0, 255, 255), 2)
        cv2.circle(vis, heel, 8, (255, 0, 0), -1)
        cv2.circle(vis, arch, 8, (0, 255, 0), -1)
        cv2.circle(vis, fore, 8, (0, 0, 255), -1)
        cv2.line(vis, heel, fore, (255, 255, 0), 2)
        cv2.line(vis, heel, arch, (255, 255, 0), 2)

        cv2.imwrite(f"{debug_dir}/side_view.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    def _save_debug_images_new(self, image, foot_mask, reference_mask, aruco_markers, 
                              measurements, calibration_data):
        """Save debug images with new calibration system"""
        debug_dir = f"output/debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(debug_dir, exist_ok=True)
        
        # Create visualization
        vis = image.copy()
        
        # Draw foot mask
        if foot_mask is not None:
            foot_overlay = np.zeros_like(vis)
            foot_overlay[foot_mask > 0] = [0, 255, 0]
            vis = cv2.addWeighted(vis, 0.7, foot_overlay, 0.3, 0)
        
        # Draw reference (card or ArUco)
        if reference_mask is not None:
            ref_overlay = np.zeros_like(vis)
            ref_overlay[reference_mask > 0] = [255, 0, 0]
            vis = cv2.addWeighted(vis, 0.7, ref_overlay, 0.3, 0)
        
        # Draw ArUco markers if detected
        if aruco_markers is not None:
            for marker_id, corners in aruco_markers.items():
                corners_int = corners.astype(int)
                cv2.polylines(vis, [corners_int], True, (0, 255, 255), 3)
                
                # Draw marker ID
                center = np.mean(corners, axis=0).astype(int)
                cv2.putText(vis, f"ID:{marker_id}", tuple(center), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Draw measurements
        if 'heel_point' in measurements and 'toe_point' in measurements:
            heel = tuple(map(int, measurements['heel_point']))
            toe = tuple(map(int, measurements['toe_point']))
            cv2.circle(vis, heel, 8, (255, 0, 0), -1)
            cv2.circle(vis, toe, 8, (0, 0, 255), -1)
            cv2.line(vis, heel, toe, (255, 255, 0), 2)
        
        # Save visualization
        cv2.imwrite(f"{debug_dir}/calibration_debug.jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # Save report
        with open(f"{debug_dir}/measurement_report.txt", 'w', encoding='utf-8') as f:
            f.write("FOOT MEASUREMENT REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Calibration Method: {calibration_data.get('calibration_method', 'unknown')}\n")
            f.write(f"ArUco Detected: {calibration_data.get('board_detected', False)}\n")
            f.write(f"3D Pose Available: {calibration_data.get('pose_info') is not None}\n")
            f.write(f"Ratio (px/mm): {calibration_data.get('ratio_px_mm', 'N/A')}\n\n")
            
            f.write("MEASUREMENTS:\n")
            for key, value in measurements.items():
                if isinstance(value, (int, float)):
                    f.write(f"- {key}: {value}\n")
        
        print(f"📁 Debug saved to: {debug_dir}")

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
        
        result = pipeline.process_foot_image(args.image, debug=args.debug)
        
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
        parser.print_help()
        
        print("\n📱 UTILISATION RAPIDE POUR MOBILE:")
        print("from mobile_sam_podiatry import quick_measure")
        print("result = quick_measure('photo_pied.jpg')")
        print("print(f\"Pied: {result[\'length_cm\']}cm x {result[\'width_cm\']}cm\")")


if __name__ == "__main__":
    main()