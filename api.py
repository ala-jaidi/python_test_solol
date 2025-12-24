
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import shutil
import os
import uuid
from mobile_sam_podiatry import MobileSAMPodiatryPipeline

app = FastAPI(
    title="Podiatry Foot Measurement API",
    description="API for measuring foot dimensions from photos for orthotic insoles.",
    version="1.0.0"
)

# Ensure output directories exist
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount output directory to serve DXF and debug images
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Initialize Pipeline (Global instance to avoid reloading model)
print("🚀 Initializing SAM Model...")
pipeline = MobileSAMPodiatryPipeline(model_type="vit_b")
print("✅ SAM Model Ready")

@app.post("/measure")
async def measure_foot(
    top_view: UploadFile = File(...),
    side_view: UploadFile = File(...),
    foot_side: str = "right"
):
    """
    Process top and side view images to generate professional foot measurements and CAD files.
    """
    try:
        # 1. Save uploaded files
        session_id = str(uuid.uuid4())
        top_ext = top_view.filename.split(".")[-1]
        side_ext = side_view.filename.split(".")[-1]
        
        top_path = os.path.join(UPLOAD_DIR, f"{session_id}_top.{top_ext}")
        side_path = os.path.join(UPLOAD_DIR, f"{session_id}_side.{side_ext}")
        
        with open(top_path, "wb") as buffer:
            shutil.copyfileobj(top_view.file, buffer)
            
        with open(side_path, "wb") as buffer:
            shutil.copyfileobj(side_view.file, buffer)
            
        # 2. Run Pipeline
        if not pipeline.initialized:
             raise HTTPException(status_code=500, detail="Model not initialized")

        result = pipeline.process_hybrid_views(
            top_path, 
            side_path, 
            debug=True, # Enable debug to save images
            foot_side=foot_side
        )
        
        # 3. Handle Errors
        if 'error' in result:
             # Cleanup uploaded files
             try:
                 os.remove(top_path)
                 os.remove(side_path)
             except:
                 pass
             return JSONResponse(status_code=400, content={"status": "error", "message": result['error']})

        # 4. Construct File URLs
        # Replace local paths with API URLs
        base_url = "/files" # This should be full domain in prod
        
        files = result.get('files', {})
        processed_files = {}
        
        for key, path in files.items():
            if path:
                # Convert absolute/relative path to URL
                # Assuming output is in OUTPUT_DIR
                rel_path = os.path.relpath(path, OUTPUT_DIR).replace("\\", "/")
                processed_files[key] = f"{base_url}/{rel_path}"
        
        result['files'] = processed_files
        
        # Add original images debug URLs if available (usually in timestamps folders)
        # For now, we return the structured result
        
        return {
            "status": "success",
            "session_id": session_id,
            "data": result
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Podiatry Measurement API is running. Use POST /measure to analyze feet."}

if __name__ == "__main__":
    import uvicorn
    # Listen on all interfaces
    uvicorn.run(app, host="0.0.0.0", port=8000)
