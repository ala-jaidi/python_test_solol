# Podiatry Foot Measurement (SAM + ArUco)
Mesures podologiques à partir de photos (vue dessus + profil) avec segmentation **Segment Anything (SAM)**, calibration **ArUco** (prioritaire) avec fallback **carte bancaire**, et export **DXF**.

## Fonctionnalités
- **Segmentation**: détection/segmentation du pied avec SAM.
- **Calibration**:
  - **ArUco L-board** (recommandé)
  - fallback **carte bancaire ISO/IEC 7810 ID-1** si ArUco non détecté
- **Mesures**:
  - vue unique: longueur, largeur, surface, périmètre, ratio L/l
  - mode hybride (2 photos): + hauteur/angle de voûte, angle des orteils
- **Exports**: images debug + **DXF** (`ezdxf`).
- **API**: FastAPI (`POST /measure`) + accès fichiers via `/files/...`.

## Installation
```bash
pip install -r requirements.txt
```

Notes:
- Le modèle SAM peut être téléchargé automatiquement au premier lancement (internet requis).
- `torch` utilisera CPU/GPU selon ta machine.

## Utilisation (CLI)
Point d’entrée: `main.py`.

### Vérifier l’installation
```bash
python main.py --validate
```

### Mesurer une image (vue unique)
```bash
python main.py path/to/photo.jpg
python main.py path/to/photo.jpg --debug
```

### Mesure hybride (vue dessus + profil)
```bash
python main.py --hybrid top.jpg side.jpg --side right
python main.py --hybrid top.jpg side.jpg --side left --debug
```

### Traitement batch
```bash
python main.py --batch path/to/folder --output results.csv
```

## Utilisation (API)
Démarrer le serveur:
```bash
uvicorn api:app --reload --port 8000
```

### Endpoint
- `POST /measure` (multipart):
  - `top_view`: image vue dessus
  - `side_view`: image profil
  - `foot_side`: `right` (défaut) ou `left`

### Fichiers générés
Les fichiers générés (DXF, images debug, etc.) sont servis via:
- `GET /files/...`

## Dossiers de sortie
- `output/`: exports DXF et images debug
- `uploads/`: images uploadées via l’API

## Structure du projet
- `mobile_sam_podiatry.py`: pipeline principal (`MobileSAMPodiatryPipeline`)
- `main.py`: CLI
- `api.py`: API FastAPI
- `dxf_export.py`: génération DXF
- `utils.py`: utilitaires
