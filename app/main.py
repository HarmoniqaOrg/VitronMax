"""
VitronMax API - a service for BBB permeability prediction.
"""
import logging
import asyncio
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from loguru import logger

from app.config import settings
from app.db import supabase
from app.models import (BatchPredictionRequest, BatchPredictionResponse, 
                     BatchPredictionStatusResponse, PredictionRequest, 
                     PredictionResponse)
from app.predict import BBBPredictor
from app.batch import BatchProcessor
from app.report import generate_pdf_report

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


@app.on_event("startup")
async def startup_event():
    """Initialize necessary resources on startup."""
    logger.info("Initializing application resources")
    
    # Ensure Supabase Storage bucket exists
    if supabase.is_configured:
        try:
            logger.info("Checking Supabase Storage bucket")
            bucket_exists = await supabase.ensure_storage_bucket_exists()
            if bucket_exists:
                logger.info("Supabase Storage bucket confirmed")
            else:
                logger.warning("Failed to confirm Supabase Storage bucket existence")
        except Exception as e:
            logger.error(f"Error checking Supabase Storage bucket: {str(e)}")
            # Don't fail startup, just log the error
            # The service can still operate with fallback to in-memory storage


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
def root() -> Dict[str, str]:
    """Root endpoint that returns service information."""
    return {"message": "Welcome to VitronMax API", "version": "1.0"}


@app.get("/healthz", tags=["health"])
async def health() -> Dict[str, str]:
    """Health check endpoint for monitoring and deployment checks."""
    return {"status": "ok"}


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
        # Create a task but await it properly in the background
        asyncio.create_task(
            supabase.store_prediction(request.smi, float(prob), predictor.version)
        )
        logger.debug(f"Initiated Supabase storage for prediction: {request.smi}")
        
        return PredictionResponse(prob=prob, version=predictor.version)
    except ValueError as exc:
        logger.warning(f"Invalid prediction request: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))


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
    
    except ValueError as exc:
        logger.warning(f"Invalid batch request: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
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
    
    except ValueError:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as exc:
        logger.error(f"Error checking job status: {str(exc)}")
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
        
        # Check if we have a signed URL from Supabase Storage
        result_url = job_status.get("result_url")
        if result_url and result_url.startswith("http"):
            logger.info(f"Redirecting to Supabase Storage URL for job {job_id}")
            # Redirect to the Supabase Storage signed URL
            return RedirectResponse(url=result_url)
            
        # Fall back to in-memory results if Supabase Storage URL is not available
        if "results" not in batch_processor.active_jobs[job_id]:
            logger.warning(f"Results not found for job {job_id}")
            raise HTTPException(status_code=404, detail="Results not found")
            
        logger.info(f"Generating CSV from in-memory results for job {job_id}")
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
        
    except ValueError:
        logger.warning(f"Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    except Exception as exc:
        logger.error(f"Error generating download: {str(exc)}")
        raise HTTPException(status_code=500, detail="Error generating download")


@app.post("/report")
async def generate_report(request: PredictionRequest):
    """Generate a PDF report for a molecule based on its SMILES string.
    
    Args:
        request: Request object containing SMILES string
        
    Returns:
        PDF file download with molecule prediction report
        
    Raises:
        HTTPException: If SMILES is invalid or prediction fails
    """
    logger.info(f"Generating PDF report for SMILES: {request.smi}")
    
    try:
        # Generate PDF report
        pdf_buffer = generate_pdf_report(request.smi)
        
        # Create a filename for the PDF
        safe_smiles = request.smi.replace("/", "_").replace("\\", "_")[:20]
        filename = f"vitronmax_report_{safe_smiles}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        # Return PDF file as a download
        return StreamingResponse(
            pdf_buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
        
    except ValueError as exc:
        logger.warning(f"Invalid SMILES for report generation: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating PDF report")
