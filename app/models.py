"""
Pydantic models for the VitronMax API.
"""
from enum import Enum
from uuid import UUID
from typing import Annotated, Dict, List, Literal, Optional
import re

from pydantic import BaseModel, Field, field_validator
from fastapi import UploadFile, File


class PredictionRequest(BaseModel):
    """Request model for BBB permeability prediction."""
    smi: str = Field(..., description="SMILES string of the molecule")

    @field_validator("smi")
    def validate_smiles(cls, v: str) -> str:
        """Validate that the input is a properly formatted SMILES string."""
        # Basic SMILES validation - this is a simplified check
        # In a real application, you might use a library like RDKit for proper validation
        if not v or not re.match(r"^[A-Za-z0-9@\-\+\[\]\(\)\\\/\%=#\.]+$", v):
            raise ValueError("Invalid SMILES string format")
        return v


class PredictionResponse(BaseModel):
    """Response model for BBB permeability prediction."""
    prob: float = Field(..., description="Probability of BBB permeability")
    version: str = Field(..., description="Model version used for prediction")


class BatchPredictionStatus(str, Enum):
    """Status of a batch prediction job."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchPredictionRequest(BaseModel):
    """Request model for starting a batch prediction."""
    filename: Optional[str] = Field(None, description="Original filename")


class BatchPredictionResponse(BaseModel):
    """Response model with batch prediction job information."""
    id: UUID = Field(..., description="Unique ID for the batch prediction job")
    status: BatchPredictionStatus = Field(..., description="Current status of the job")
    filename: Optional[str] = Field(None, description="Original filename")
    total_molecules: int = Field(0, description="Total number of molecules to process")
    processed_molecules: int = Field(0, description="Number of processed molecules")
    created_at: str = Field(..., description="Timestamp when the job was created")
    completed_at: Optional[str] = Field(None, description="Timestamp when the job completed")
    result_url: Optional[str] = Field(None, description="URL to download the results")
    

class BatchPredictionStatusResponse(BaseModel):
    """Response model for checking a batch prediction status."""
    id: UUID = Field(..., description="Unique ID for the batch prediction job")
    status: BatchPredictionStatus = Field(..., description="Current status of the job")
    progress: float = Field(..., description="Progress as a percentage (0-100)")
    filename: Optional[str] = Field(None, description="Original filename")
    created_at: str = Field(..., description="Timestamp when the job was created")
    result_url: Optional[str] = Field(None, description="URL to download the results if available")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
