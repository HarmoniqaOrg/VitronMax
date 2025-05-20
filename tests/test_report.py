"""
Tests for PDF report generation functionality.
"""
import io
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


from app.main import app
from app.report import PDFReportGenerator
from app.predict import BBBPredictor


client = TestClient(app)


def test_report_endpoint_valid_smiles():
    """Test that the /report endpoint returns a PDF for valid SMILES."""
    # Setup - valid SMILES
    test_smiles = "CCO"  # ethanol
    
    # Mock the BBBPredictor to avoid model loading issues
    with patch('app.report.BBBPredictor') as MockPredictor:
        # Configure the mocks
        mock_instance = MockPredictor.return_value
        mock_instance.predict.return_value = 0.75
        mock_instance.version = "1.0"
        
        # Make the request
        response = client.post("/report", json={"smi": test_smiles})
        
        # Assert response
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert "attachment; filename=vitronmax_report" in response.headers["content-disposition"]
        assert response.content.startswith(b"%PDF")
        


def test_report_endpoint_invalid_smiles():
    """Test that the /report endpoint returns an error for invalid SMILES."""
    # Setup - invalid SMILES
    test_smiles = "invalid_smiles_123"
        
    # Mock the BBBPredictor to avoid model loading issues
    with patch('app.report.BBBPredictor') as MockPredictor:
        # Configure the mock to raise an error for invalid SMILES
        mock_instance = MockPredictor.return_value
        mock_instance.predict.side_effect = ValueError("Invalid SMILES")
        
        # Make the request
        response = client.post("/report", json={"smi": test_smiles})
            
        # Assert - it could be 422 (validation) or 400 (application error)
        # depending on how FastAPI processes the error
        assert response.status_code in (400, 422)
        # Response body may contain validation error or detailed error message
        response_json = response.json()
        assert "invalid" in str(response_json).lower() or "error" in str(response_json).lower()


def test_report_endpoint_empty_smiles():
    """Test that the /report endpoint returns 422 for empty SMILES."""
    # Make the request with empty SMILES
    response = client.post("/report", json={"smi": ""})
        
    # Assert
    assert response.status_code == 422
    # FastAPI validation errors are returned as a list of errors
    error_response = response.json()
    assert isinstance(error_response, dict) or isinstance(error_response, list)
    # Make sure there's a validation error mentioned somewhere in the error
    assert "error" in str(error_response).lower() or "value" in str(error_response).lower()


def test_pdf_report_generator():
    """Test the PDF report generator directly."""
    # Setup
    test_smiles = "CCO"  # ethanol
    
    # Mock the BBBPredictor.predict method to return a known value
    with patch.object(BBBPredictor, "predict", return_value=0.75):
        # Create the generator
        generator = PDFReportGenerator(test_smiles)
        
        # Generate the report
        pdf_buffer = generator.generate_report()
        
        # Check if we got a BytesIO object
        assert isinstance(pdf_buffer, io.BytesIO)
        
        # Check if it contains PDF data
        assert pdf_buffer.getvalue().startswith(b"%PDF")


def test_pdf_report_generator_sections():
    """Test that the PDF report generator creates all required sections."""
    # Setup
    test_smiles = "CCO"  # ethanol
    
    # Mock the necessary functions
    with patch.object(BBBPredictor, "predict", return_value=0.75):
        # Create the generator
        generator = PDFReportGenerator(test_smiles)
        
        # Test header section
        header_elements = generator._create_header()
        assert len(header_elements) > 0
        
        # Test molecule section
        generator.generate_prediction_data()
        molecule_elements = generator._create_molecule_section()
        assert len(molecule_elements) > 0
        
        # Test prediction section
        prediction_elements = generator._create_prediction_section()
        assert len(prediction_elements) > 0
        
        # Test footer section
        footer_elements = generator._create_footer()
        assert len(footer_elements) > 0


def test_interpretation_thresholds():
    """Test that the interpretation text uses appropriate thresholds."""
    # Setup
    test_smiles = "CCO"  # ethanol
    generator = PDFReportGenerator(test_smiles)
    
    # Test different probability interpretations
    assert "Very likely" in generator._get_interpretation(0.95)
    assert "Likely" in generator._get_interpretation(0.75)
    assert "Uncertain" in generator._get_interpretation(0.5)
    assert "Unlikely" in generator._get_interpretation(0.2)
    assert "Very unlikely" in generator._get_interpretation(0.05)
