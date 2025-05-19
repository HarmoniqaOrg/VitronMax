"""
Pydantic models for the VitronMax API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Annotated, Literal
import re


class PredictionRequest(BaseModel):
    """Request model for BBB permeability prediction."""
    smi: str = Field(..., description="SMILES string of the molecule")

    @field_validator("smi")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        """Validate that the input is a properly formatted SMILES string."""
        # Basic SMILES validation - this is a simplified check
        # In a real application, you might use a library like RDKit for proper validation
        if not v or not re.match(r"^[A-Za-z0-9@\-\+\[\]\(\)\\\/%=#\.]+$", v):
            raise ValueError("Invalid SMILES string format")
        return v


class PredictionResponse(BaseModel):
    """Response model for BBB permeability prediction."""
    prob: float = Field(..., description="Probability of BBB permeability")
    version: str = Field(..., description="Model version used for prediction")
