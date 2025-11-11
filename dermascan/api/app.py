"""
FastAPI Application for DermaScan

Main API server that handles:
- Image upload
- Model inference
- Result delivery
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from dermascan.inference.predictor import DermaScanPredictor
from dermascan.preprocessing.image_processor import ImageProcessor
from dermascan.database.conditions import SkinConditionDatabase

app = FastAPI(
    title="DermaScan API",
    description="AI-powered dermatological diagnosis system",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

# Initialize components
predictor = DermaScanPredictor()
processor = ImageProcessor()
db = SkinConditionDatabase()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main frontend page"""
    with open("frontend/templates/index.html", "r") as f:
        return f.read()


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "0.1.0"}


@app.post("/api/predict")
async def predict_condition(file: UploadFile = File(...)):
    """
    Predict skin condition from uploaded image

    Args:
        file: Uploaded image file

    Returns:
        JSON with predictions and condition information
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")

        # Read and process image
        contents = await file.read()
        processed_image = processor.process_uploaded_image(contents)

        # Get predictions
        predictions = predictor.predict(processed_image)

        # Get top 3 predictions with details
        results = []
        for pred in predictions[:3]:
            condition_info = db.get_condition_info(pred['class_name'])
            results.append({
                "condition": pred['class_name'],
                "confidence": float(pred['confidence']),
                "description": condition_info['description'],
                "severity": condition_info['severity'],
                "recommendations": condition_info['recommendations']
            })

        return JSONResponse(content={
            "success": True,
            "predictions": results,
            "warning": "This is not a medical diagnosis. Please consult a healthcare professional."
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/conditions")
async def list_conditions():
    """Get list of all supported skin conditions"""
    return {"conditions": db.list_all_conditions()}


@app.get("/api/conditions/{condition_name}")
async def get_condition_details(condition_name: str):
    """Get detailed information about a specific condition"""
    info = db.get_condition_info(condition_name)
    if info:
        return info
    raise HTTPException(status_code=404, detail="Condition not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
