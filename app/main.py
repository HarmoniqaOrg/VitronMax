"""
VitronMax API - a service for BBB permeability prediction.
"""
import logging
from pathlib import Path
from typing import Dict, Union, Any

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger

from app.config import settings
from app.db import supabase
from app.models import (BatchPredictionRequest, BatchPredictionResponse, 
                     BatchPredictionStatusResponse, PredictionRequest, 
                     PredictionResponse)
from app.predict import BBBPredictor
from app.batch import BatchProcessor

# Configure logging based on environment
log_level = getattr(logging, settings.LOG_LEVEL)
logger.remove()
logger.add(
    "stderr",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
)

logger.info(f"Starting VitronMax in {settings.ENV.value} environment")

app = FastAPI(
    title="VitronMax",
    description="API for blood-brain barrier permeability prediction",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred"},
    )

# Initialize the predictor at startup to avoid loading the model on every request
predictor = BBBPredictor()

# Initialize the batch processor
batch_processor = BatchProcessor(predictor)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint that returns service information."""
    logger.debug("Root endpoint called")
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
    logger.info(f"Processing prediction request for SMILES: {request.smi}")
    try:
        # Validation already done by pydantic
        prob = predictor.predict(request.smi)
        logger.info(f"Prediction for {request.smi}: {prob:.4f}")
        
        # Store the prediction in Supabase
        # This is non-blocking, so it won't delay the response
        supabase_task = supabase.store_prediction(request.smi, float(prob), predictor.version)
        # We don't await this task, allowing it to run in the background
        logger.debug(f"Initiated Supabase storage for prediction: {request.smi}")
        
        return PredictionResponse(prob=prob, version=predictor.version)
    except ValueError as e:
        logger.warning(f"Invalid prediction request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch_predict_csv", response_model=BatchPredictionResponse, status_code=202)
async def batch_predict_csv(
    file: UploadFile = File(...),
    batch_request: BatchPredictionRequest = Depends(),
) -> BatchPredictionResponse:
    """Process a batch of SMILES from a CSV file and return predictions.
    
    The CSV file must have a header row with a column named 'SMILES' or 'smi'.
    This endpoint starts an asynchronous job. Use the returned job ID to check status.
    
    Args:
        file: CSV file with SMILES strings
        batch_request: Additional request parameters
        
    Returns:
        Job ID and initial status information
        
    Raises:
        HTTPException: If the file format is invalid
    """
    logger.info(f"Received batch prediction request with file: {file.filename}")
    
    try:
        # Start a new batch job
        job_id = await batch_processor.start_batch_job(file)
        
        # Get initial job status
        job_status = batch_processor.get_job_status(job_id)
        
        return BatchPredictionResponse(
            id=job_status["id"],
            status=job_status["status"],
            filename=job_status["filename"],
            total_molecules=job_status["total_molecules"] if "total_molecules" in job_status else 0,
            processed_molecules=job_status["processed_molecules"] if "processed_molecules" in job_status else 0,
            created_at=job_status["created_at"],
            result_url=None  # Results not available immediately
        )
    
    except ValueError as e:
        logger.warning(f"Invalid batch request: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing batch request: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing batch request")


@app.get("/batch_status/{job_id}", response_model=BatchPredictionStatusResponse)
async def get_batch_status(job_id: str) -> BatchPredictionStatusResponse:
    """Get the status of a batch prediction job.
    
    Args:
        job_id: The job ID to check
        
    Returns:
        Current job status information
        
    Raises:
        HTTPException: If the job is not found
    """
    logger.info(f"Checking status for batch job: {job_id}")
    
    try:
        job_status = batch_processor.get_job_status(job_id)
        
        return BatchPredictionStatusResponse(
            id=job_status["id"],
            status=job_status["status"],
            progress=job_status["progress"],
            filename=job_status["filename"],
            created_at=job_status["created_at"],
            result_url=job_status["result_url"],
            error_message=job_status.get("error_message")
        )
    
    except ValueError as e:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        raise HTTPException(status_code=500, detail="Error checking job status")


@app.get("/download/{job_id}")
async def download_results(job_id: str):
    """Download the results of a batch prediction job as a CSV file.
    
    Args:
        job_id: The job ID of the completed batch job
        
    Returns:
        CSV file with prediction results
        
    Raises:
        HTTPException: If the job is not found or not completed
    """
    logger.info(f"Download request for batch job: {job_id}")
    
    try:
        job_status = batch_processor.get_job_status(job_id)
        
        # Check if job is completed
        if job_status["status"] != "completed":
            logger.warning(f"Cannot download results for job {job_id}: Job not completed")
            raise HTTPException(
                status_code=400, 
                detail=f"Cannot download results for incomplete job. Status: {job_status['status']}"
            )
        
        # In a real implementation, this might fetch from a cloud storage service
        # For this MVP, we're generating it from the in-memory results
        if "results" not in batch_processor.active_jobs[job_id]:
            logger.warning(f"Results not found for job {job_id}")
            raise HTTPException(status_code=404, detail="Results not found")
            
        results = batch_processor.active_jobs[job_id]["results"]
        csv_content = batch_processor._generate_results_csv(results)
        
        # Create a filename based on the original filename or job ID
        filename = job_status["filename"]
        if not filename or not filename.endswith(".csv"):
            filename = f"predictions_{job_id}.csv"
        
        # Return CSV file as a download
        return StreamingResponse(
            iter([csv_content]), 
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except ValueError as e:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as e:
        logger.error(f"Error generating download: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating download")
