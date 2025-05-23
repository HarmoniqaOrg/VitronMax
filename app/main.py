"""
VitronMax API - a service for BBB permeability prediction.
"""

import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, AsyncGenerator, Union
from contextlib import asynccontextmanager
import uvicorn
from fastapi.responses import JSONResponse, StreamingResponse, RedirectResponse
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    File,
    status,  # Add status import
    BackgroundTasks,  # Add BackgroundTasks import
)
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.db import supabase
from app.models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    BatchPredictionStatus,  # Import BatchPredictionStatus enum
)
from app.predict import BBBPredictor
from app.batch import BatchProcessor, JobDetailsDict  # ADDED JobDetailsDict here
from app.report import generate_pdf_report
from uuid import UUID

# Configure logging based on environment
log_level = getattr(logging, settings.LOG_LEVEL)
logger.remove()
logger.add(
    "stderr",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=log_level,
)

logger.info(f"Starting VitronMax in {settings.ENV.value} environment")


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events."""
    logger.info("Initializing application resources (lifespan startup)")

    # Ensure Supabase Storage bucket exists
    if supabase.is_configured:
        try:
            logger.info("Checking Supabase Storage bucket (lifespan)")
            bucket_exists = await supabase.ensure_storage_bucket()
            if bucket_exists:
                logger.info("Supabase Storage bucket confirmed (lifespan)")
            else:
                logger.warning(
                    "Failed to confirm Supabase Storage bucket existence (lifespan)"
                )
        except Exception as e:
            logger.error(f"Error checking Supabase Storage bucket (lifespan): {str(e)}")
            # Don't fail startup, just log the error

    yield

    # Shutdown logic would go here, if any
    logger.info("Application shutdown (lifespan)")


app = FastAPI(
    title="VitronMax",
    description="API for blood-brain barrier permeability prediction",
    version="1.0.0",
    lifespan=lifespan,  # Pass lifespan to the constructor
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
batch_processor = BatchProcessor(predictor=predictor, supabase_client=supabase)

# Fixed ID for debugging this specific scenario
DEBUG_FIXED_JOB_ID = "123e4567-e89b-12d3-a456-426614174000"  # Valid UUID


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def root(request: Request) -> RedirectResponse:
    # Redirect root to /docs for API documentation
    return RedirectResponse(url="/docs")


@app.get(
    "/api/v1/batch/job_status/{job_id}",
    response_model=JobDetailsDict,  # CORRECTED response_model
    tags=["batch"],
)
def get_batch_job_status(job_id: str) -> JobDetailsDict:  # REMOVED async
    """Get the status and details of a batch prediction job."""
    logger.info(f"Fetching status for job_id: {job_id}")
    try:
        # BatchProcessor.get_job_status is synchronous
        job_details = batch_processor.get_job_status(job_id)
        if (
            not job_details
        ):  # Should not happen if get_job_status raises ValueError for not found
            raise ValueError(f"Job ID '{job_id}' not found.")  # Defensive
        return job_details
    except ValueError as e:
        logger.warning(f"ValueError when fetching status for job_id {job_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        )  # CORRECTED detail
    except Exception as e:
        logger.error(
            f"Unexpected error fetching status for job_id {job_id}: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while fetching job status.",
        )


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


@app.post(
    "/api/v1/batch/predict_csv",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["batch"],
    summary="Submit a CSV file for batch BBB permeability prediction",
    description="Accepts a CSV file containing SMILES strings, processes them asynchronously, "
    "and returns a job ID for status tracking and result retrieval.",
    name="batch_predict_csv",
)
async def submit_batch_predict_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> BatchPredictionResponse:
    logger.info(f"submit_batch_predict_csv called for file: {file.filename}")
    logger.debug(f"BackgroundTasks instance ID: {id(background_tasks)}")

    if not file.filename:  # REINTRODUCED filename check
        raise HTTPException(status_code=400, detail="File name is required.")

    try:
        # Call start_batch_job to get the actual job_id
        job_id = await batch_processor.start_batch_job(file=file)  # REINTRODUCED
        logger.info(f"Batch job started with ID: {job_id}")
    except ValueError as e:
        logger.error(f"ValueError during batch job start: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Use the actual job_id for the background task
    logger.debug(f"About to call background_tasks.add_task for job_id: {job_id}")
    background_tasks.add_task(batch_processor.process_batch_job, job_id)

    # For the response, get the status using the new job_id
    response_data: JobDetailsDict = batch_processor.get_job_status(job_id)

    # Construct BatchPredictionResponse directly
    return BatchPredictionResponse(
        id=UUID(response_data["id"]),  # Convert str to UUID
        status=BatchPredictionStatus(response_data["status"]),  # Convert str to enum
        filename=response_data.get("filename"),
        total_molecules=response_data["total_molecules"],
        processed_molecules=response_data["processed_molecules"],
        created_at=response_data["created_at"],
        completed_at=response_data.get("completed_at"),
        result_url=response_data.get("result_url"),
    )


@app.get(
    "/api/v1/batch/results/{job_id}/download",
    tags=["batch"],
    response_model=None,  # Added to prevent FastAPI error
)
async def download_results(job_id: str) -> Union[RedirectResponse, StreamingResponse]:
    """Download the results of a batch prediction job as a CSV file.

    Args:
        job_id: The job ID of the completed batch job

    Returns:
        CSV file with prediction results

    Raises:
        HTTPException: If the job is not found or not completed
    """
    logger.info(f"[DIAG] Received download request for job ID: {job_id}")
    try:
        job_details = batch_processor.get_job_status(job_id)
        logger.info(f"[DIAG] download_results - job_details retrieved: {job_details}")
    except KeyError:
        logger.warning(f"[DIAG] Job ID {job_id} not found for download (KeyError).")
        raise HTTPException(status_code=404, detail=f"Job ID {job_id} not found")

    logger.info(f"[DIAG] download_results - job status: {job_details['status']}")
    logger.info(
        f"[DIAG] download_results - expected completed status: {BatchPredictionStatus.COMPLETED.value}"
    )
    if job_details["status"] != BatchPredictionStatus.COMPLETED.value:
        logger.warning(
            f"[DIAG] Job ID {job_id} not completed. Current status: {job_details['status']}"
        )
        raise HTTPException(
            status_code=404,  # Or 400/409 depending on desired semantics
            detail="Job not completed or results not available yet.",
        )

    result_url_from_details = job_details.get("result_url")
    supabase_configured_status = supabase.is_configured
    logger.info(f"[DIAG] download_results - result_url: {result_url_from_details}")
    logger.info(
        f"[DIAG] download_results - supabase.is_configured: {supabase_configured_status}"
    )

    # Check if a result_url (e.g., from Supabase Storage) is available
    if result_url_from_details and supabase_configured_status:
        logger.info(
            f"[DIAG] Redirecting to Supabase Storage URL for job {job_id}: {result_url_from_details}"
        )
        # Client should handle this redirect to download the file directly from storage
        # This assumes result_url is a direct, presigned download link
        return RedirectResponse(url=result_url_from_details)

    # Fallback: Generate CSV on the fly if no result_url or Supabase not configured
    logger.info(f"[DIAG] Generating CSV results on-the-fly for job {job_id}")
    try:
        if not job_details.get("results"):
            logger.error(f"Job {job_id} has no results to generate CSV from.")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job has no results available to generate a CSV file.",
            )
        # Ensure job_details["results"] is List[ResultItemDict]
        results_data = job_details.get("results", [])
        csv_content = batch_processor._generate_results_csv(results_data)

        original_filename = job_details.get("filename")
        if original_filename:
            base_filename = os.path.splitext(original_filename)[0]
        else:
            # Use job_id for a default filename if original is not available
            base_filename = f"job_{job_id}"
        response_filename = f"{base_filename}_results.csv"

        return StreamingResponse(
            iter([csv_content.encode()]),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{response_filename}"'
            },
        )
    except Exception as e:
        logger.error(f"Error generating/streaming CSV for job {job_id}: {str(e)}")
        logger.exception(f"Full traceback for CSV generation error job {job_id}:")
        raise HTTPException(status_code=500, detail="Error generating result CSV file.")


@app.post("/report", response_class=StreamingResponse, tags=["report"])
async def generate_report(request: PredictionRequest) -> StreamingResponse:
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
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except ValueError as exc:
        logger.warning(f"Invalid SMILES for report generation: {str(exc)}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating PDF report")


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.SERVER_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
