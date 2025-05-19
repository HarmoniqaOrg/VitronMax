"""Supabase integration for VitronMax."""
from typing import Dict, Optional, Any, List
import uuid

import httpx
from loguru import logger

from app.config import settings
from app.models import BatchPredictionStatus


class SupabaseClient:
    """Simple client for Supabase interactions."""

    def __init__(self) -> None:
        """Initialize the Supabase client."""
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_SERVICE_KEY
        self.is_configured = bool(self.url and self.key)
        
        if not self.is_configured:
            logger.warning("Supabase is not configured. Database operations will be skipped.")
        else:
            logger.info(f"Supabase client initialized with URL: {self.url}")
    
    async def store_prediction(self, smiles: str, probability: float, model_version: str = "v1.0") -> Optional[Dict[str, Any]]:
        """Store a prediction in the Supabase database.
        
        Args:
            smiles: The SMILES string that was predicted
            probability: The prediction probability
            model_version: Version of the model used for prediction
            
        Returns:
            The Supabase response data or None if not configured
        """
        if not self.is_configured:
            logger.debug("Skipping database storage - Supabase not configured")
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "smiles": smiles,
                    "probability": probability,
                    "model_version": model_version
                }
                
                logger.debug(f"Storing prediction in Supabase: {data}")
                response = await client.post(
                    f"{self.url}/rest/v1/predictions",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201):
                    logger.info(f"Successfully stored prediction for {smiles}")
                    return response.json()
                else:
                    logger.error(f"Failed to store prediction: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error storing prediction: {e}")
            return None
    
    async def create_batch_job(self, job_id: str, filename: Optional[str], total_molecules: int) -> Optional[Dict[str, Any]]:
        """Create a new batch prediction job in Supabase.
        
        Args:
            job_id: UUID of the batch job
            filename: Original filename (optional)
            total_molecules: Total number of molecules to process
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            logger.debug("Skipping database storage - Supabase not configured")
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "id": job_id,
                    "status": BatchPredictionStatus.PENDING.value,
                    "filename": filename,
                    "total_molecules": total_molecules
                }
                
                logger.debug(f"Creating batch job in Supabase: {data}")
                response = await client.post(
                    f"{self.url}/rest/v1/batch_predictions",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201):
                    logger.info(f"Successfully created batch job {job_id}")
                    return response.json()
                else:
                    logger.error(f"Failed to create batch job: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error creating batch job: {e}")
            return None
    
    async def update_batch_job_status(self, job_id: str, status: str) -> Optional[Dict[str, Any]]:
        """Update the status of a batch job.
        
        Args:
            job_id: UUID of the batch job
            status: New status
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "status": status
                }
                
                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201, 204):
                    logger.debug(f"Updated batch job {job_id} status to {status}")
                    return {"success": True}
                else:
                    logger.error(f"Failed to update batch job status: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error updating batch job status: {e}")
            return None
    
    async def update_batch_job_progress(self, job_id: str, processed_molecules: int) -> Optional[Dict[str, Any]]:
        """Update the progress of a batch job.
        
        Args:
            job_id: UUID of the batch job
            processed_molecules: Number of processed molecules
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "processed_molecules": processed_molecules
                }
                
                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201, 204):
                    return {"success": True}
                else:
                    logger.error(f"Failed to update batch job progress: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error updating batch job progress: {e}")
            return None
            
    async def store_batch_prediction_item(self, batch_id: str, smiles: str, 
                                        row_number: int, model_version: str,
                                        probability: Optional[float] = None, 
                                        error_message: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Store a single prediction item in a batch job.
        
        Args:
            batch_id: UUID of the batch job
            smiles: SMILES string
            row_number: Row number in the original CSV
            model_version: Model version used
            probability: Prediction probability (None if error)
            error_message: Error message if the prediction failed
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "batch_id": batch_id,
                    "smiles": smiles,
                    "row_number": row_number,
                    "model_version": model_version
                }
                
                if probability is not None:
                    data["probability"] = probability
                    
                if error_message:
                    data["error_message"] = error_message
                
                response = await client.post(
                    f"{self.url}/rest/v1/batch_prediction_items",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code not in (200, 201):
                    logger.error(f"Failed to store batch prediction item: {response.status_code} {response.text}")
                
                return None if response.status_code not in (200, 201) else {"success": True}
                    
        except Exception as e:
            logger.exception(f"Error storing batch prediction item: {e}")
            return None
            
    async def complete_batch_job(self, job_id: str, result_url: str) -> Optional[Dict[str, Any]]:
        """Mark a batch job as completed.
        
        Args:
            job_id: UUID of the batch job
            result_url: URL to download the results
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "status": BatchPredictionStatus.COMPLETED.value,
                    "completed_at": "now()",
                    "result_url": result_url
                }
                
                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201, 204):
                    logger.info(f"Batch job {job_id} marked as completed")
                    return {"success": True}
                else:
                    logger.error(f"Failed to complete batch job: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error completing batch job: {e}")
            return None
            
    async def fail_batch_job(self, job_id: str, error_message: str) -> Optional[Dict[str, Any]]:
        """Mark a batch job as failed.
        
        Args:
            job_id: UUID of the batch job
            error_message: Error message
            
        Returns:
            The Supabase response or None if not configured
        """
        if not self.is_configured:
            return None
            
        try:
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal"
                }
                
                data = {
                    "status": BatchPredictionStatus.FAILED.value,
                    "error_message": error_message,
                    "completed_at": "now()"
                }
                
                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0
                )
                
                if response.status_code in (200, 201, 204):
                    logger.info(f"Batch job {job_id} marked as failed: {error_message}")
                    return {"success": True}
                else:
                    logger.error(f"Failed to mark batch job as failed: {response.status_code} {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Error marking batch job as failed: {e}")
            return None


# Create a global instance
supabase = SupabaseClient()
