"""Supabase integration for VitronMax."""

from typing import Dict, Optional, Any, Mapping, cast

import httpx
from loguru import logger

from app.config import settings
from app.models import BatchPredictionStatus

# Constants for Supabase Storage
STORAGE_BATCH_RESULTS_PATH = "batch_results"
URL_EXPIRY_SECONDS = 60 * 60 * 24 * 7  # 7 days


class SupabaseClient:
    """Simple client for Supabase interactions."""

    def __init__(self) -> None:
        """Initialize the Supabase client."""
        self.url = settings.SUPABASE_URL
        self.key = settings.SUPABASE_SERVICE_KEY
        self.is_configured = bool(self.url and self.key)

        if not self.is_configured:
            logger.warning(
                "Supabase is not configured. Database operations will be skipped."
            )
        else:
            logger.info(f"Supabase client initialized with URL: {self.url}")

    async def ensure_storage_bucket_exists(self) -> bool:
        """Ensure that the Supabase Storage bucket exists.

        This method checks if the storage bucket exists and creates it if it doesn't.

        Returns:
            bool: True if the bucket exists or was created successfully, False otherwise
        """
        if not self.is_configured:
            logger.debug("Skipping storage bucket check - Supabase not configured")
            return False

        bucket_name = settings.STORAGE_BUCKET_NAME

        try:
            # First check if the bucket exists
            async with httpx.AsyncClient() as client:
                headers: Mapping[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                }

                # Get list of buckets
                response = await client.get(
                    f"{self.url}/storage/v1/bucket", headers=headers, timeout=10.0
                )

                if response.status_code == 200:
                    buckets = response.json()
                    bucket_exists = any(
                        bucket["name"] == bucket_name for bucket in buckets
                    )

                    if bucket_exists:
                        logger.info(f"Storage bucket '{bucket_name}' already exists")
                        return True

                    # Bucket doesn't exist, create it
                    logger.info(
                        f"Storage bucket '{bucket_name}' not found, creating it..."
                    )
                    create_response = await client.post(
                        f"{self.url}/storage/v1/bucket",
                        headers=headers,
                        json={
                            "name": bucket_name,
                            "public": False,  # Private bucket for security
                            "file_size_limit": 5242880,  # 5MB limit for CSV files
                        },
                        timeout=10.0,
                    )

                    if create_response.status_code in (200, 201):
                        logger.info(
                            f"Successfully created storage bucket '{bucket_name}'"
                        )
                        return True
                    else:
                        logger.error(
                            f"Failed to create storage bucket: {create_response.text}"
                        )
                        return False
                else:
                    logger.error(f"Failed to list storage buckets: {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error checking/creating storage bucket: {str(e)}")
            return False

    async def store_prediction(
        self, smiles: str, probability: float, model_version: str = "v1.0"
    ) -> Optional[Dict[str, Any]]:
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
                headers: Dict[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }

                data = {
                    "smiles": smiles,
                    "probability": probability,
                    "model_version": model_version,
                }

                logger.debug(f"Storing prediction in Supabase: {data}")
                response = await client.post(
                    f"{self.url}/rest/v1/predictions",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201):
                    logger.info(f"Successfully stored prediction for {smiles}")
                    return cast(Dict[str, Any], response.json())
                else:
                    logger.error(
                        f"Failed to store prediction: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error storing prediction: {e}")
            return None

    async def create_batch_job(
        self, job_id: str, filename: Optional[str], total_molecules: int
    ) -> Optional[Dict[str, Any]]:
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
                headers: Dict[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }

                data = {
                    "id": job_id,
                    "status": BatchPredictionStatus.PENDING.value,
                    "filename": filename,
                    "total_molecules": total_molecules,
                }

                logger.debug(f"Creating batch job in Supabase: {data}")
                response = await client.post(
                    f"{self.url}/rest/v1/batch_predictions",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201):
                    logger.info(f"Successfully created batch job {job_id}")
                    return cast(Dict[str, Any], response.json())
                else:
                    logger.error(
                        f"Failed to create batch job: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error creating batch job: {e}")
            return None

    async def update_batch_job_status(
        self, job_id: str, status: str
    ) -> Optional[Dict[str, Any]]:
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
                headers: Dict[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }

                data = {"status": status}

                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201, 204):
                    logger.debug(f"Updated batch job {job_id} status to {status}")
                    return {"success": True}
                else:
                    logger.error(
                        f"Failed to update batch job status: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error updating batch job status: {e}")
            return None

    async def update_batch_job_progress(
        self, job_id: str, processed_molecules: int
    ) -> Optional[Dict[str, Any]]:
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
                headers: Dict[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }

                data = {"processed_molecules": processed_molecules}

                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201, 204):
                    return {"success": True}
                else:
                    logger.error(
                        f"Failed to update batch job progress: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error updating batch job progress: {e}")
            return None

    async def store_batch_prediction_item(
        self,
        batch_id: str,
        smiles: str,
        row_number: int,
        model_version: str,
        probability: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
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
                headers: Dict[str, str] = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                }

                data = {
                    "batch_id": batch_id,
                    "smiles": smiles,
                    "row_number": row_number,
                    "model_version": model_version,
                }

                if probability is not None:
                    data["probability"] = probability

                if error_message:
                    data["error_message"] = error_message

                response = await client.post(
                    f"{self.url}/rest/v1/batch_prediction_items",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code not in (200, 201):
                    logger.error(
                        f"Failed to store batch prediction item: {response.status_code} {response.text}"
                    )
                    return cast(str, response.text)

                return {"success": True}

        except Exception as e:
            logger.exception(f"Error storing batch prediction item: {e}")
            return None

    async def complete_batch_job(
        self, job_id: str, result_url: str
    ) -> Optional[Dict[str, Any]]:
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
                    "Prefer": "return=minimal",
                }

                data = {
                    "status": BatchPredictionStatus.COMPLETED.value,
                    "completed_at": "now()",
                    "result_url": result_url,
                }

                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201, 204):
                    logger.info(f"Batch job {job_id} marked as completed")
                    return {"success": True}
                else:
                    logger.error(
                        f"Failed to complete batch job: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error completing batch job: {e}")
            return None

    async def fail_batch_job(
        self, job_id: str, error_message: str
    ) -> Optional[Dict[str, Any]]:
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
                    "Prefer": "return=minimal",
                }

                data = {
                    "status": BatchPredictionStatus.FAILED.value,
                    "error_message": error_message,
                    "completed_at": "now()",
                }

                response = await client.patch(
                    f"{self.url}/rest/v1/batch_predictions?id=eq.{job_id}",
                    headers=headers,
                    json=data,
                    timeout=5.0,
                )

                if response.status_code in (200, 201, 204):
                    logger.info(f"Batch job {job_id} marked as failed: {error_message}")
                    return {"success": True}
                else:
                    logger.error(
                        f"Failed to mark batch job as failed: {response.status_code} {response.text}"
                    )
                    return None

        except Exception as e:
            logger.exception(f"Error marking batch job as failed: {e}")
            return None

    async def ensure_storage_bucket(self) -> bool:
        """Ensure the storage bucket exists for batch results.

        Creates the bucket if it doesn't exist.

        Returns:
            True if the bucket exists or was created, False otherwise
        """
        # This method is deprecated in favor of ensure_storage_bucket_exists
        # Kept for backwards compatibility
        return await self.ensure_storage_bucket_exists()

    async def store_batch_result_csv(
        self, job_id: str, csv_content: str
    ) -> Optional[str]:
        """Store batch results in Supabase Storage and return a signed URL.

        Args:
            job_id: UUID of the batch job
            csv_content: CSV content as string

        Returns:
            Signed URL for accessing the file or None if storage failed
        """
        if not self.is_configured:
            logger.debug("Skipping batch result storage - Supabase not configured")
            return None

        try:
            # Ensure storage bucket exists
            bucket_exists = await self.ensure_storage_bucket()
            if not bucket_exists:
                logger.error("Failed to ensure storage bucket exists")
                return None

            # File path in storage
            file_path = f"{STORAGE_BATCH_RESULTS_PATH}/{job_id}.csv"

            # Upload CSV content
            async with httpx.AsyncClient() as client:
                headers = {
                    "apikey": self.key,
                    "Authorization": f"Bearer {self.key}",
                    "Content-Type": "text/csv",
                }

                # Convert CSV string to bytes
                csv_bytes = csv_content.encode("utf-8")

                bucket_name = settings.STORAGE_BUCKET_NAME

                # Upload the file
                upload_response = await client.post(
                    f"{self.url}/storage/v1/object/{bucket_name}/{file_path}",
                    headers=headers,
                    content=csv_bytes,
                    timeout=10.0,  # Longer timeout for file upload
                )

                if upload_response.status_code not in (200, 201):
                    logger.error(
                        f"Failed to upload batch results: {upload_response.status_code} {upload_response.text}"
                    )
                    return None

                # Generate signed URL for the file
                signed_url_params = {"expiresIn": URL_EXPIRY_SECONDS}

                bucket_name = settings.STORAGE_BUCKET_NAME

                signed_url_response = await client.post(
                    f"{self.url}/storage/v1/object/sign/{bucket_name}/{file_path}",
                    headers={
                        "apikey": self.key,
                        "Authorization": f"Bearer {self.key}",
                        "Content-Type": "application/json",
                    },
                    json=signed_url_params,
                    timeout=5.0,
                )

                if signed_url_response.status_code in (200, 201):
                    result = signed_url_response.json()
                    signed_url = result.get("signedURL")
                    if signed_url:
                        # If URL doesn't include the base, prepend it
                        if not signed_url.startswith("http"):
                            signed_url = f"{self.url}{signed_url}"
                        logger.info(f"Generated signed URL for batch results {job_id}")
                        return signed_url

                logger.error(
                    f"Failed to generate signed URL: {signed_url_response.status_code} {signed_url_response.text}"
                )
                return None

        except Exception as e:
            logger.exception(f"Error storing batch results: {e}")
            return None


# Create a global instance
supabase = SupabaseClient()
