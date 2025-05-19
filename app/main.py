"""
VitronMax API - a service for BBB permeability prediction.
"""
from fastapi import FastAPI, HTTPException
from pathlib import Path

from app.models import PredictionRequest, PredictionResponse
from app.predict import BBBPredictor

app = FastAPI(
    title="VitronMax",
    description="API for blood-brain barrier permeability prediction",
    version="1.0.0",
)

# Initialize the predictor at startup to avoid loading the model on every request
predictor = BBBPredictor()


@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint that returns service information."""
    return {"message": "Welcome to VitronMax API", "version": "1.0"}


@app.post("/predict_fp", response_model=PredictionResponse)
async def predict_fp(request: PredictionRequest) -> PredictionResponse:
    """Predict blood-brain barrier permeability using fingerprints.
    
    Args:
        request: Request object containing SMILES string
        
    Returns:
        Prediction probability and model version
        
    Raises:
        HTTPException: If SMILES is invalid or prediction fails
    """
    try:
        # Validation already done by pydantic
        prob = predictor.predict(request.smi)
        return PredictionResponse(prob=prob, version=predictor.version)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
