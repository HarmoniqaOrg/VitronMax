"""
Module for handling batch prediction operations for CSV files.
"""
import csv
import asyncio
import uuid
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger
from fastapi import UploadFile

from app.models import BatchPredictionStatus
from app.predict import BBBPredictor
from app.db import supabase


class BatchProcessor:
    """Handler for batch prediction processing."""

    def __init__(self, predictor: BBBPredictor):
        """Initialize the batch processor.
        
        Args:
            predictor: The prediction model to use
        """
        self.predictor = predictor
        self.active_jobs: Dict[str, Dict] = {}
    
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
            text = contents.decode('utf-8-sig')
            reader = csv.reader(io.StringIO(text))
            
            # Check if file is empty
            header = next(reader, None)
            if not header:
                return False, "Empty CSV file", []
            
            # Find SMILES column
            smiles_col = None
            for i, col in enumerate(header):
                if col.lower() in ('smiles', 'smi', 'smile'):
                    smiles_col = i
                    break
            
            if smiles_col is None:
                return False, "Could not find SMILES column in CSV header", []
            
            # Extract SMILES from the CSV
            smiles_list = []
            for i, row in enumerate(reader):
                if i >= 1000:  # Limit to 1000 molecules per batch
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
        
        # Create job in database
        if supabase.is_configured:
            # Insert record into batch_predictions table
            logger.info(f"Creating batch job in database: {job_id}")
            await supabase.create_batch_job(
                job_id=job_id,
                filename=filename,
                total_molecules=len(smiles_list)
            )
        
        # Store job info in memory (for demo purposes)
        self.active_jobs[job_id] = {
            "id": job_id,
            "status": BatchPredictionStatus.PENDING.value,
            "filename": filename,
            "total_molecules": len(smiles_list),
            "processed_molecules": 0,
            "created_at": datetime.now().isoformat(),
            "completed_at": None,
            "smiles_list": smiles_list,
            "result_url": None,
            "results": []
        }
        
        # Start the prediction job in the background
        asyncio.create_task(self.process_batch_job(job_id))
        
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
        if supabase.is_configured:
            await supabase.update_batch_job_status(job_id, BatchPredictionStatus.PROCESSING.value)
        
        smiles_list = job["smiles_list"]
        results = []
        
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
                        "error": None
                    }
                    
                    # Insert into database if configured
                    if supabase.is_configured:
                        await supabase.store_batch_prediction_item(
                            batch_id=job_id,
                            smiles=smi,
                            probability=float(prob),
                            model_version=self.predictor.version,
                            row_number=i
                        )
                    
                except ValueError as e:
                    # Handle invalid SMILES
                    result = {
                        "smiles": smi,
                        "probability": None,
                        "model_version": self.predictor.version,
                        "error": str(e)
                    }
                    
                    # Store error in database
                    if supabase.is_configured:
                        await supabase.store_batch_prediction_item(
                            batch_id=job_id,
                            smiles=smi,
                            probability=None,
                            model_version=self.predictor.version,
                            row_number=i,
                            error_message=str(e)
                        )
                
                # Add to results
                results.append(result)
                
                # Update progress
                job["processed_molecules"] = i + 1
                
                # Update progress in database
                if supabase.is_configured and (i + 1) % 10 == 0:  # Update every 10 molecules
                    await supabase.update_batch_job_progress(job_id, i + 1)
                
                # Small delay to prevent overloading (and simulate processing time)
                await asyncio.sleep(0.01)
            
            # Generate results CSV
            csv_content = self._generate_results_csv(results)
            
            # Store results in Supabase Storage
            result_url = None
            if supabase.is_configured:
                # Store CSV in Supabase Storage and get signed URL
                logger.info(f"Storing batch results in Supabase Storage for job {job_id}")
                result_url = await supabase.store_batch_result_csv(job_id, csv_content)
                
                if not result_url:
                    logger.warning(f"Failed to store batch results in Supabase Storage for job {job_id}")
                    # Fallback to in-memory storage for MVP
                    result_url = f"/download/{job_id}"
            else:
                # Fallback to in-memory storage if Supabase is not configured
                result_url = f"/download/{job_id}"
            
            # Save results in memory as fallback
            job["results"] = results
            
            # Update job status to COMPLETED
            job["status"] = BatchPredictionStatus.COMPLETED.value
            job["completed_at"] = datetime.now().isoformat()
            job["result_url"] = result_url
            
            # Update database
            if supabase.is_configured:
                await supabase.complete_batch_job(
                    job_id=job_id,
                    result_url=result_url
                )
            
            logger.info(f"Batch job {job_id} completed successfully")
            
        except Exception as e:
            # Handle any unexpected errors
            logger.error(f"Error processing batch job {job_id}: {str(e)}")
            
            # Update job status to FAILED
            job["status"] = BatchPredictionStatus.FAILED.value
            job["error_message"] = str(e)
            
            # Update database
            if supabase.is_configured:
                await supabase.fail_batch_job(
                    job_id=job_id,
                    error_message=str(e)
                )
    
    def get_job_status(self, job_id: str) -> Dict:
        """Get the status of a batch prediction job.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Job status information
            
        Raises:
            ValueError: If the job is not found
        """
        if job_id not in self.active_jobs:
            raise ValueError(f"Job {job_id} not found")
        
        job = self.active_jobs[job_id]
        
        # Calculate progress percentage
        total = job["total_molecules"]
        processed = job["processed_molecules"]
        progress = (processed / total * 100) if total > 0 else 0
        
        return {
            "id": job["id"],
            "status": job["status"],
            "progress": progress,
            "filename": job["filename"],
            "created_at": job["created_at"],
            "completed_at": job["completed_at"],
            "result_url": job["result_url"],
            "error_message": job.get("error_message")
        }
    
    @staticmethod
    def _generate_results_csv(results: List[Dict]) -> str:
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
            writer.writerow([
                result["smiles"],
                result["probability"] if result["probability"] is not None else "",
                result["model_version"],
                result["error"] if result["error"] else ""
            ])
        
        return output.getvalue()
