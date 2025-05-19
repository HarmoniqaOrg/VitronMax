"""
Tests for the prediction API endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_predict_fp_valid_smiles():
    """Test prediction with valid SMILES string."""
    response = client.post(
        "/predict_fp",
        json={"smi": "CCO"}  # Ethanol SMILES
    )
    assert response.status_code == 200
    data = response.json()
    assert "prob" in data
    assert isinstance(data["prob"], float)
    assert 0 <= data["prob"] <= 1
    assert data["version"] == "1.0"


def test_predict_fp_invalid_smiles():
    """Test prediction with invalid SMILES string."""
    response = client.post(
        "/predict_fp",
        json={"smi": "invalid!!!"}  # Invalid SMILES
    )
    assert response.status_code == 400
    assert "Invalid SMILES" in response.json()["detail"]


def test_predict_fp_empty_smiles():
    """Test prediction with empty SMILES string."""
    response = client.post(
        "/predict_fp",
        json={"smi": ""}  # Empty SMILES
    )
    assert response.status_code == 422  # Validation error
