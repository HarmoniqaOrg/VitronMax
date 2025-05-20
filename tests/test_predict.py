"""
Tests for the prediction API endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@pytest.mark.parametrize("_", [(None)])
def test_predict_fp_valid_smiles(_):
    """Test prediction with valid SMILES string."""
    response = client.post("/predict_fp", json={"smi": "CCO"})  # Ethanol SMILES
    assert response.status_code == 200
    data = response.json()
    assert "prob" in data
    assert isinstance(data["prob"], float)
    assert 0 <= data["prob"] <= 1
    assert "version" in data


def test_predict_fp_invalid_smiles():
    """Test prediction with invalid SMILES string."""
    response = client.post("/predict_fp", json={"smi": "invalid!!!"})  # Invalid SMILES
    # FastAPI may return 400 (our explicit validation) or 422 (Pydantic validation)
    assert response.status_code in (400, 422)
    response_json = response.json()
    assert "detail" in response_json


def test_predict_fp_empty_smiles():
    """Test prediction with empty SMILES string."""
    response = client.post("/predict_fp", json={"smi": ""})  # Empty SMILES
    assert response.status_code == 422  # FastAPI validation error
