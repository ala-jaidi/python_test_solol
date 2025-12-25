from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import shutil
import os
import uuid
from mobile_sam_podiatry import MobileSAMPodiatryPipeline

app = FastAPI(
    title="Podiatry Foot Measurement API",
    description="API for measuring foot dimensions from photos for Flutter mobile app.",
    version="2.0.0"
)

# CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure output directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount output directory to serve debug images
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Initialize Pipeline (Global instance to avoid reloading model)
print("🚀 Initializing SAM Model...")
pipeline = MobileSAMPodiatryPipeline(model_type="vit_b")
print("✅ SAM Model Ready")


# ============== FLUTTER MOBILE ROUTES ==============

@app.post("/measure/top")
async def measure_top_view(
    image: UploadFile = File(...),
    foot_side: str = Form("right")
):
    """
    📐 Vue dessus - Mesure largeur + toe_angle
    
    Params:
    - image: Photo vue dessus du pied avec ArUco L-board visible
    - foot_side: "left" ou "right"
    
    Returns: width_cm, toe_angle_deg
    """
    try:
        session_id = str(uuid.uuid4())
        ext = image.filename.split(".")[-1]
        image_path = os.path.join(UPLOAD_DIR, f"{session_id}_top_{foot_side}.{ext}")
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        if not pipeline.initialized:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        result = pipeline.process_top_view(image_path, debug=True, foot_side=foot_side)
        
        if 'error' in result:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": result['error'],
                "foot_side": foot_side
            })
        
        # Trouver le dossier debug le plus récent
        import glob
        debug_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "debug_*")), reverse=True)
        debug_image_url = None
        if debug_dirs:
            latest_debug = os.path.basename(debug_dirs[0])
            debug_image_url = f"/files/{latest_debug}/calibration_debug.jpg"
        
        # URL du fichier DXF
        dxf_url = None
        if 'dxf_path' in result and result['dxf_path']:
            dxf_filename = os.path.basename(result['dxf_path'])
            dxf_url = f"/files/{dxf_filename}"

        return {
            "success": True,
            "foot_side": foot_side,
            "view": "top",
            "width_cm": result['width_cm'],
            "toe_angle_deg": result.get('toe_angle_deg', 0), # Optional now
            "toe_width_cm": result.get('toe_width_cm', 0),
            "debug_image_url": debug_image_url,
            "dxf_url": dxf_url,
            "session_id": session_id
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/measure/side")
async def measure_side_view(
    image: UploadFile = File(...),
    foot_side: str = Form("right")
):
    """
    📏 Vue profil - Mesure longueur heel↔toe avec points et image debug
    
    Params:
    - image: Photo vue profil du pied avec ArUco L-board visible
    - foot_side: "left" ou "right"
    
    Returns: length_cm, heel_point, toe_point, debug_image_url (image avec points dessinés)
    """
    try:
        session_id = str(uuid.uuid4())
        ext = image.filename.split(".")[-1]
        image_path = os.path.join(UPLOAD_DIR, f"{session_id}_side_{foot_side}.{ext}")
        
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        if not pipeline.initialized:
            raise HTTPException(status_code=500, detail="Model not initialized")
        
        # Utilise process_side_view pour avoir longueur uniquement
        result = pipeline.process_side_view(image_path, debug=True, foot_side=foot_side)
        
        if 'error' in result:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": result['error'],
                "foot_side": foot_side
            })
        
        # Trouver le dossier debug le plus récent
        import glob
        debug_dirs = sorted(glob.glob(os.path.join(OUTPUT_DIR, "debug_*")), reverse=True)
        debug_image_url = None
        if debug_dirs:
            latest_debug = os.path.basename(debug_dirs[0])
            debug_image_url = f"/files/{latest_debug}/calibration_debug.jpg"
        
        # URL du fichier DXF
        dxf_url = None
        if 'dxf_path' in result and result['dxf_path']:
            dxf_filename = os.path.basename(result['dxf_path'])
            dxf_url = f"/files/{dxf_filename}"
        
        return {
            "success": True,
            "foot_side": foot_side,
            "view": "side",
            "length_cm": result['length_cm'],
            "width_cm": 0, # Side view ignores width
            "heel_point": result.get('heel_point'),
            "toe_point": result.get('toe_point'),
            "confidence": result.get('confidence', 0),
            "aruco_detected": result.get('aruco_detected', False),
            "debug_image_url": debug_image_url,
            "dxf_url": dxf_url,
            "session_id": session_id
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


class FootMeasurement(BaseModel):
    width_cm: float
    length_cm: float
    toe_angle_deg: Optional[float] = None
    toe_width_cm: Optional[float] = None


class CompleteMeasurementRequest(BaseModel):
    left_foot: Optional[FootMeasurement] = None
    right_foot: Optional[FootMeasurement] = None


@app.post("/measure/complete")
async def complete_measurement(data: CompleteMeasurementRequest):
    """
    ✅ Résultats finaux - Reçoit les mesures des deux pieds
    
    Body JSON:
    {
        "left_foot": {"width_cm": 9.5, "length_cm": 26.0, "toe_angle_deg": 15.0},
        "right_foot": {"width_cm": 9.3, "length_cm": 25.8, "toe_angle_deg": 14.5}
    }
    
    Returns: Résumé des mesures + recommandation pointure
    """
    try:
        result = {
            "success": True,
            "measurements": {}
        }
        
        if data.left_foot:
            result["measurements"]["left"] = {
                "width_cm": data.left_foot.width_cm,
                "length_cm": data.left_foot.length_cm,
                "toe_angle_deg": data.left_foot.toe_angle_deg,
                "shoe_size_eu": _length_to_eu_size(data.left_foot.length_cm)
            }
        
        if data.right_foot:
            result["measurements"]["right"] = {
                "width_cm": data.right_foot.width_cm,
                "length_cm": data.right_foot.length_cm,
                "toe_angle_deg": data.right_foot.toe_angle_deg,
                "shoe_size_eu": _length_to_eu_size(data.right_foot.length_cm)
            }
        
        # Recommandation basée sur le pied le plus grand
        lengths = []
        if data.left_foot:
            lengths.append(data.left_foot.length_cm)
        if data.right_foot:
            lengths.append(data.right_foot.length_cm)
        
        if lengths:
            max_length = max(lengths)
            result["recommended_shoe_size_eu"] = _length_to_eu_size(max_length)
            result["max_length_cm"] = max_length
        
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _length_to_eu_size(length_cm: float) -> int:
    """Convertit la longueur du pied en pointure EU"""
    # Formule standard: EU = (longueur_cm + 1.5) * 1.5
    return round((length_cm + 1.5) * 1.5)

@app.get("/")
def read_root():
    return {"message": "Podiatry Measurement API is running. Use POST /measure/side or /measure/top."}

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)
