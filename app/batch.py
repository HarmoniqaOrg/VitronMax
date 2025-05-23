"""
Module for handling batch prediction operations for CSV files.
"""

import csv
import asyncio
import uuid
import io
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dateutil import tz
from typing_extensions import TypedDict

from loguru import logger
from fastapi import UploadFile

from app.models import BatchPredictionStatus
from app.predict import BBBPredictor

SMILES_MAX_COUNT = 10000  # Maximum number of SMILES strings to process in a batch


# Define TypedDict for job details structure used in active_jobs
class ResultItemDict(TypedDict, total=False):
    smiles: str
    probability: Optional[float]
    model_version: Optional[str]
    error: Optional[str]


class JobDetailsDict(TypedDict):
    id: str
    status: str
    filename: Optional[str]
    total_molecules: int
    processed_molecules: int
    progress: float
    created_at: str
    completed_at: Optional[str]
    smiles_list: List[str]
    result_url: Optional[str]
    error_message: Optional[str]
    results: List[ResultItemDict]


class BatchProcessor:
    """Handler for batch prediction processing."""

    def __init__(self, predictor: BBBPredictor, supabase_client: Any):
        """Initialize the batch processor.

        Args:
            predictor: The prediction model to use
            supabase_client: The Supabase client instance
        """
        self.predictor = predictor
        self.active_jobs: Dict[str, JobDetailsDict] = {}
        self.supabase = supabase_client

    @staticmethod
    async def validate_csv(file: UploadFile) -> Tuple[bool, Optional[str], List[str]]:
        """Validate that the uploaded CSV file contains valid SMILES.

        Args:
            file: The uploaded CSV file

        Returns:
            Tuple containing (is_valid, error_message, list_of_smiles)
        """
        try:
            # Read the file contents
            contents = await file.read()
            file.file.seek(0)  # Reset file pointer

            # Try to parse as CSV
            text = contents.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text))

            # Check if file is empty
            header = next(reader, None)
            if not header:
                return False, "Empty CSV file", []

            # Find SMILES column
            smiles_col = None
            for i, col in enumerate(header):
                if col.lower() in ("smiles", "smi", "smile"):
                    smiles_col = i
                    break

            if smiles_col is None:
                return False, "Could not find SMILES column in CSV header", []

            # Extract SMILES from the CSV
            smiles_list = []
            for i, row in enumerate(reader):
                if i >= SMILES_MAX_COUNT:  # Limit to 1000 molecules per batch
                    logger.warning(
                        f"CSV contains more than {SMILES_MAX_COUNT} molecules. Processing only the first {SMILES_MAX_COUNT}."
                    )
                    break

                if len(row) > smiles_col:
                    smi = row[smiles_col].strip()
                    if smi:  # Skip empty SMILES
                        smiles_list.append(smi)

            if not smiles_list:
                return False, "No valid SMILES found in the CSV file", []

            return True, None, smiles_list

        except UnicodeDecodeError:
            return False, "File is not a valid CSV (encoding error)", []
        except csv.Error:
            return False, "File is not a valid CSV (parsing error)", []
        except Exception as e:
            logger.error(f"Error validating CSV: {str(e)}")
            return False, f"Error processing CSV file: {str(e)}", []

    async def start_batch_job(self, file: UploadFile) -> str:
        """Start a new batch prediction job.

        Args:
            file: CSV file with SMILES strings

        Returns:
            Job ID

        Raises:
            ValueError: If the file is invalid
        """
        # Validate the CSV file
        is_valid, error_msg, smiles_list = await self.validate_csv(file)
        if not is_valid:
            raise ValueError(error_msg)

        # Create a new job ID
        job_id = str(uuid.uuid4())

        # Original filename
        filename = file.filename
        current_time_iso = datetime.now(tz.tzutc()).isoformat()
        current_status = BatchPredictionStatus.PENDING.value

        # Create job in database
        if self.supabase.is_configured:
            # Insert record into batch_predictions table
            logger.info(f"Creating batch job in database: {job_id}")
            await self.supabase.create_batch_job(
                job_id=job_id,
                filename=filename,
                total_molecules=len(smiles_list),
            )

        # Store job info in memory (for demo purposes)
        self.active_jobs[job_id] = {
            "id": job_id,
            "status": current_status,
            "filename": filename,
            "total_molecules": len(smiles_list),
            "processed_molecules": 0,
            "progress": 0.0,
            "created_at": current_time_iso,
            "completed_at": None,
            "smiles_list": smiles_list,
            "result_url": None,
            "error_message": None,
            "results": [],
        }

        # Start the prediction job in the background - This will be handled by FastAPI's BackgroundTasks in the route
        # asyncio.create_task(self.process_batch_job(job_id)) # Removed this line

        return job_id

    async def process_batch_job(self, job_id: str) -> None:
        """Process a batch prediction job.

        Args:
            job_id: The job ID to process
        """
        if job_id not in self.active_jobs:
            logger.error(f"Job {job_id} not found")
            return

        # Update job status to PROCESSING
        job = self.active_jobs[job_id]
        job["status"] = BatchPredictionStatus.PROCESSING.value

        # Update status in database if available
        if self.supabase.is_configured:
            await self.supabase.update_batch_job_status(
                job_id, BatchPredictionStatus.PROCESSING  # Use enum member
            )

        smiles_list = job["smiles_list"]
        job_results_list: List[ResultItemDict] = []  # Use a local list to build results
        result: ResultItemDict

        try:
            # Process each SMILES
            for i, smi in enumerate(smiles_list):
                try:
                    # Make prediction
                    prob = self.predictor.predict(smi)

                    # Store result
                    result = {
                        "smiles": smi,
                        "probability": float(prob),
                        "model_version": self.predictor.version,
                        "error": None,
                    }

                    # Insert into database if configured
                    if self.supabase.is_configured:
                        await self.supabase.store_batch_prediction_item(
                            batch_id=job_id,
                            smiles=smi,
                            prediction_result=result.get("probability"),
                            model_version=result.get("model_version"),
                            error_message=result.get("error"),
                        )

                except ValueError as ve:
                    logger.warning(f"Validation error for SMILES '{smi}': {ve}")
                    result = {
                        "smiles": smi,
                        "probability": None,
                        "model_version": self.predictor.version,  # or None if error is before prediction
                        "error": str(ve),
                    }

                    # Store error in database
                    if self.supabase.is_configured:
                        await self.supabase.store_batch_prediction_item(
                            batch_id=job_id,
                            smiles=smi,
                            prediction_result=None,
                            model_version=self.predictor.version,
                            error_message=str(ve),
                        )

                except Exception as e:
                    logger.error(f"Error predicting SMILES '{smi}': {e}")
                    result = {
                        "smiles": smi,
                        "probability": None,
                        "model_version": self.predictor.version,  # or None if error is before prediction
                        "error": f"Prediction error: {str(e)}",
                    }

                job_results_list.append(result)
                job["processed_molecules"] = i + 1

                # Update progress in database
                if (
                    self.supabase.is_configured and (i + 1) % 10 == 0
                ):  # Update every 10 molecules
                    await self.supabase.update_batch_job_progress(job_id, i + 1)

                # Small delay to prevent overloading (and simulate processing time)
                await asyncio.sleep(0.01)

            # All SMILES processed, mark as COMPLETED (locally first)
            job["status"] = BatchPredictionStatus.COMPLETED.value
            job["completed_at"] = datetime.now(tz.tzutc()).isoformat()
            job["results"] = job_results_list  # Assign the fully populated list

            # Generate CSV and upload to Supabase if configured
            if self.supabase.is_configured:
                csv_content = self._generate_results_csv(job_results_list)
                result_url = await self.supabase.store_batch_result_csv(
                    csv_content=csv_content, job_id=job_id, filename=job["filename"]
                )
                job["result_url"] = result_url
                await self.supabase.complete_batch_job(
                    job_id=job_id,
                    status=BatchPredictionStatus.COMPLETED,  # Use enum member
                    result_url=result_url,
                    error_message=None,
                )

        except Exception as e:
            logger.error(f"Error processing batch job {job_id}: {e}")
            job["status"] = BatchPredictionStatus.FAILED.value
            job["error_message"] = str(e)
            job["completed_at"] = datetime.now(tz.tzutc()).isoformat()
            # Ensure results processed so far are still in job["results"]
            # If the error happened during store_batch_result_csv, job_results_list is complete.
            # If it happened during individual SMILES processing, job_results_list contains results up to the error.
            job["results"] = job_results_list  # Assign whatever was processed

            if self.supabase.is_configured:
                await self.supabase.complete_batch_job(
                    job_id=job_id,
                    status=BatchPredictionStatus.FAILED,  # Use enum member
                    result_url=None,
                    error_message=str(e),
                )

        logger.info(f"Finished processing batch job {job_id}")

    def get_job_status(self, job_id: str) -> JobDetailsDict:
        """Get the status of a batch prediction job.

        Args:
            job_id: The job ID to check

        Returns:
            Job status information

        Raises:
            ValueError: If the job is not found
        """
        if job_id not in self.active_jobs:
            # This path is for jobs not found in the in-memory dict.
            # For jobs found via db_client.get_batch_job, different logic applies.
            # The current mypy error is for this in-memory path.
            raise ValueError(f"Job {job_id} not found in active jobs")

        job = self.active_jobs[job_id]

        # Calculate progress percentage
        total = job["total_molecules"]
        processed = job["processed_molecules"]
        progress_val = (processed / total * 100.0) if total > 0 else 0.0

        # Ensure all keys from JobDetailsDict are present
        details_to_return: JobDetailsDict = {
            "id": job["id"],
            "status": job["status"],
            "filename": job.get("filename"),  # filename is Optional in JobDetailsDict
            "total_molecules": job["total_molecules"],
            "processed_molecules": job["processed_molecules"],
            "progress": progress_val,
            "created_at": job["created_at"],
            "completed_at": job.get("completed_at"),  # completed_at is Optional
            "smiles_list": job["smiles_list"],
            "result_url": job.get("result_url"),  # result_url is Optional
            "error_message": job.get("error_message"),  # error_message is Optional
            "results": job["results"],
        }
        return details_to_return

    @staticmethod
    def _generate_results_csv(results: List[ResultItemDict]) -> str:
        """Generate a CSV string from prediction results.

        Args:
            results: List of prediction results

        Returns:
            CSV content as a string
        """
        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(["SMILES", "BBB_Probability", "Model_Version", "Error"])

        # Write data
        for result in results:
            writer.writerow(
                [
                    result["smiles"],
                    result["probability"] if result["probability"] is not None else "",
                    result["model_version"],
                    result["error"] if result["error"] else "",
                ]
            )

        return output.getvalue()
