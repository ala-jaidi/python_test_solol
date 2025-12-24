
import ezdxf
import numpy as np
import cv2

class DXFExporter:
    """
    Module professionnel pour l'exportation de contours de pieds en DXF (AutoCAD).
    Utilisé pour la fabrication de semelles orthopédiques (CAO/CFAO).
    """
    
    @staticmethod
    def export_contour_to_dxf(contour, ratio_px_mm, output_path, filename="foot_contour.dxf"):
        """
        Convertit un contour OpenCV en fichier DXF à l'échelle réelle (mm).
        
        Args:
            contour: Contour OpenCV (numpy array de points)
            ratio_px_mm: Ratio de conversion pixels vers millimètres
            output_path: Dossier de sortie
            filename: Nom du fichier
        
        Returns:
            str: Chemin complet du fichier généré
        """
        import os
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        full_path = os.path.join(output_path, filename)
        
        # Créer un nouveau document DXF (R2010 est très compatible)
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # 1. Simplifier légèrement le contour pour réduire le nombre de points (lissage)
        # epsilon = 0.5px (très fin pour garder la précision)
        epsilon = 0.001 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # 2. Convertir les points en millimètres
        # OpenCV contour shape is (N, 1, 2) -> we need (N, 2)
        points_px = approx_contour[:, 0, :]
        
        # Y-flip: En image (0,0) est en haut à gauche. En CAD, Y monte.
        # On inverse Y pour avoir le pied dans le bon sens.
        points_mm = []
        
        # Centrer le pied à l'origine (0,0) pour faciliter l'import CAO
        center_x = np.mean(points_px[:, 0])
        center_y = np.mean(points_px[:, 1])
        
        for pt in points_px:
            x_mm = (pt[0] - center_x) / ratio_px_mm
            # Inversion Y et conversion
            y_mm = -(pt[1] - center_y) / ratio_px_mm 
            points_mm.append((x_mm, y_mm))
        
        # Fermer la boucle
        if len(points_mm) > 0:
            points_mm.append(points_mm[0])
            
        # 3. Dessiner la polyligne
        msp.add_lwpolyline(points_mm, dxfattribs={'layer': 'FOOT_CONTOUR', 'color': 1}) # Rouge
        
        # Ajouter des métadonnées ou repères (optionnel)
        # Axe central
        msp.add_line((0, -100), (0, 100), dxfattribs={'layer': 'AXIS', 'color': 2}) # Jaune
        
        # Sauvegarder
        try:
            doc.saveas(full_path)
            print(f"✅ DXF généré avec succès: {full_path}")
            return full_path
        except Exception as e:
            print(f"❌ Erreur lors de la génération DXF: {e}")
            return None

    @staticmethod
    def create_side_view_profile_dxf(profile_points, ratio_px_mm, output_path, filename="side_profile.dxf"):
        """
        Exporte la courbe de profil (voûte plantaire) en DXF.
        """
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        full_path = os.path.join(output_path, filename)
        
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()
        
        # Conversion pixels -> mm
        points_mm = []
        if len(profile_points) > 0:
            # Normaliser par rapport au sol (y = 0)
            # Supposons que profile_points est [(x, y_height_from_ground), ...]
            
            # On centre X
            xs = [p[0] for p in profile_points]
            center_x = np.mean(xs) if xs else 0
            
            for pt in profile_points:
                x_mm = (pt[0] - center_x) / ratio_px_mm
                y_mm = pt[1] / ratio_px_mm # Hauteur en mm
                points_mm.append((x_mm, y_mm))
                
        msp.add_spline(points_mm, dxfattribs={'layer': 'ARCH_PROFILE', 'color': 3}) # Vert
        
        doc.saveas(full_path)
        return full_path
