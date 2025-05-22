"""
Unit tests for the BatchProcessor class methods (minimal external dependencies).
"""

import pytest
from io import BytesIO
from typing import Dict, Any, List

from unittest.mock import AsyncMock
from typing import cast
from fastapi import UploadFile
from pytest_mock import MockerFixture

from app.models import (
    BatchPredictionStatus,
)
from app.batch import BatchProcessor, SMILES_MAX_COUNT, ResultItemDict


# Tests for BatchProcessor.validate_csv
@pytest.mark.asyncio
async def test_validate_csv_valid(
    processor_with_mock_predictor: BatchProcessor, mock_upload_file_valid: UploadFile
) -> None:
    """Test validating a valid CSV file."""
    processor = processor_with_mock_predictor
    is_valid, error_msg, smiles_list = await processor.validate_csv(
        mock_upload_file_valid
    )
    assert is_valid is True
    assert error_msg is None
    assert smiles_list == ["CCO", "CCC"]  # Corrected based on conftest.py


@pytest.mark.asyncio
async def test_validate_csv_invalid_no_smiles_column(
    processor_with_mock_predictor: BatchProcessor,
) -> None:
    """Test validate_csv with CSV missing SMILES column."""
    # Create a CSV file content without 'SMILES' column
    csv_content = "ID,Name\n1,Ethanol\n2,Benzene"
    file = UploadFile(filename="test_no_smiles.csv", file=BytesIO(csv_content.encode()))
    is_valid, error_msg, smiles_list = await BatchProcessor.validate_csv(file)
    assert is_valid is False
    assert error_msg == "Could not find SMILES column in CSV header"
    assert smiles_list == []


@pytest.mark.asyncio
async def test_validate_csv_invalid_empty(
    processor_with_mock_predictor: BatchProcessor,
) -> None:
    """Test validate_csv with an empty CSV file and header-only CSV."""
    processor = processor_with_mock_predictor
    # Test with a truly empty CSV file content
    csv_content_empty = ""
    file_empty = UploadFile(
        filename="empty.csv", file=BytesIO(csv_content_empty.encode())
    )
    is_valid, error_msg, smiles_list = await processor.validate_csv(file_empty)
    assert is_valid is False
    assert error_msg == "Empty CSV file"
    assert smiles_list == []

    # Test with only header
    csv_content_header_only = "SMILES\n"
    file_header_only = UploadFile(
        filename="header_only.csv", file=BytesIO(csv_content_header_only.encode())
    )
    is_valid, error_msg, smiles_list = await processor.validate_csv(file_header_only)
    assert is_valid is False
    assert error_msg == "No valid SMILES found in the CSV file"
    assert smiles_list == []


@pytest.mark.asyncio
async def test_validate_csv_smiles_limit(
    processor_with_mock_predictor: BatchProcessor,
) -> None:
    """Test validate_csv with a CSV file exceeding SMILES_MAX_COUNT."""
    smiles_count = SMILES_MAX_COUNT + 1
    csv_lines = ["SMILES"] + [f"C{i}" for i in range(smiles_count)]
    csv_content = "\n".join(csv_lines)
    file = UploadFile(
        filename="too_many_smiles.csv", file=BytesIO(csv_content.encode())
    )

    is_valid, error_msg, smiles_list = await BatchProcessor.validate_csv(file)
    assert is_valid is True  # File is still valid, just truncated
    assert error_msg is None
    assert len(smiles_list) == SMILES_MAX_COUNT


# Tests for _update_job_status (via its callers or direct tests if made public)
# @pytest.mark.asyncio
# async def test_update_job_status_supabase_not_configured(
#     processor_with_mock_predictor: BatchProcessor, mock_supabase_client_not_configured: MagicMock,
# ) -> None:
#     """Test updating job status when Supabase is not configured."""
#     processor = processor_with_mock_predictor
#     processor.supabase = mock_supabase_client_not_configured
#
#     job_id = "test_job_123"
#     status = BatchPredictionStatus.PROCESSING
#     processor.active_jobs[job_id] = {
#         "id": job_id,
#         "status": BatchPredictionStatus.PENDING.value,
#         "smiles_list": ["CCO"],
#     }
#
#     await processor._update_job_status(job_id, status)
#
#     assert processor.active_jobs[job_id]["status"] == status.value
#     # Verify Supabase client's update_batch_job was NOT called
#     mock_supabase_client_not_configured.update_batch_job_status.assert_not_called()
#     mock_supabase_client_not_configured.complete_batch_job.assert_not_called()
#     mock_supabase_client_not_configured.fail_batch_job.assert_not_called()


# @pytest.mark.asyncio
# async def test_update_job_status_supabase_configured_success(
#     processor_with_mock_predictor: BatchProcessor,
#     mock_supabase_client_configured: MagicMock,
# ) -> None:
#     """Test updating job status to COMPLETED with Supabase configured."""
#     processor = processor_with_mock_predictor
#     processor.supabase = mock_supabase_client_configured
#
#     job_id = "test_job_456"
#     status = BatchPredictionStatus.COMPLETED
#     results_data = [{"smiles": "CCO", "probability": 0.8, "model_version": "1.0"}]
#     result_url = "http://example.com/results.csv"
#
#     processor.active_jobs[job_id] = {
#         "id": job_id,
#         "status": BatchPredictionStatus.PROCESSING.value,
#         "smiles_list": ["CCO"],
#         "results": [],
#         "result_url": None,
#     }
#
#     await processor._update_job_status(
#         job_id, status, results=results_data, result_url=result_url
#     )
#
#     assert processor.active_jobs[job_id]["status"] == status.value
#     assert processor.active_jobs[job_id]["results"] == results_data
#     assert processor.active_jobs[job_id]["result_url"] == result_url
#     mock_supabase_client_configured.complete_batch_job.assert_called_once_with(
#         job_id, result_url
#     )
#     mock_supabase_client_configured.update_batch_job_status.assert_not_called()
#     mock_supabase_client_configured.fail_batch_job.assert_not_called()


# @pytest.mark.asyncio
# async def test_update_job_status_supabase_configured_failure(
#     processor_with_mock_predictor: BatchProcessor,
#     mock_supabase_client_configured: MagicMock,
# ) -> None:
#     """Test updating job status to FAILED with Supabase configured."""
#     processor = processor_with_mock_predictor
#     processor.supabase = mock_supabase_client_configured
#
#     job_id = "test_job_789"
#     status = BatchPredictionStatus.FAILED
#     error_message = "Something went wrong"
#
#     processor.active_jobs[job_id] = {
#         "id": job_id,
#         "status": BatchPredictionStatus.PROCESSING.value,
#         "smiles_list": ["CCO"],
#         "error_message": None,
#         "result_url": None,
#     }
#
#     await processor._update_job_status(
#         job_id, status, error_message=error_message
#     )
#
#     assert processor.active_jobs[job_id]["status"] == status.value
#     assert processor.active_jobs[job_id]["error_message"] == error_message
#     mock_supabase_client_configured.fail_batch_job.assert_called_once_with(
#         job_id, error_message
#     )
#     mock_supabase_client_configured.update_batch_job_status.assert_not_called()
#     mock_supabase_client_configured.complete_batch_job.assert_not_called()


@pytest.mark.asyncio
async def test_start_batch_job_valid_csv(
    processor_with_mock_predictor: BatchProcessor,
    mock_upload_file_valid: UploadFile,
    mocker: MockerFixture,
) -> None:
    """Test starting a batch job with a valid CSV, mocking _parse_csv and process_batch_job."""
    processor = processor_with_mock_predictor
    # processor.supabase_client is already a MagicMock via processor_with_mock_predictor fixture
    processor.supabase.is_configured = True
    processor.supabase.create_batch_job = AsyncMock(return_value=None)

    parsed_filename = mock_upload_file_valid.filename
    parsed_smiles_list = ["CCO", "CNC"]
    mock_parse_csv = mocker.patch.object(
        processor,
        "validate_csv",
        new_callable=AsyncMock,
        return_value=(True, None, parsed_smiles_list),
    )

    job_id = await processor.start_batch_job(mock_upload_file_valid)

    assert job_id is not None
    assert job_id in processor.active_jobs
    job_details = processor.active_jobs[job_id]
    assert job_details["filename"] == parsed_filename
    assert job_details["status"] == BatchPredictionStatus.PENDING.value
    assert job_details["total_molecules"] == len(parsed_smiles_list)
    assert job_details["smiles_list"] == parsed_smiles_list

    mock_parse_csv.assert_called_once_with(mock_upload_file_valid)

    # Check if Supabase create_batch_job was called if configured
    processor.supabase.create_batch_job.assert_called_once_with(
        job_id=job_id,
        filename=mock_upload_file_valid.filename,
        total_molecules=len(parsed_smiles_list),
    )


# Test with a mix of successful and error results
@pytest.mark.asyncio
async def test_generate_results_csv(
    processor_with_mock_predictor: BatchProcessor,
) -> None:
    """Test generating CSV content from prediction results."""
    job_details: Dict[str, Any] = {
        "id": "test_job_123",
        "filename": "test.csv",
        "status": BatchPredictionStatus.COMPLETED.value,
        "results": [
            {
                "smiles": "CCO",
                "probability": 0.8,
                "model_version": "1.0",
                "error": None,
            },
            {
                "smiles": "invalid",
                "probability": None,
                "model_version": None,
                "error": "Invalid SMILES",
            },
            {
                "smiles": "CCC",
                "probability": 0.3,
                "model_version": "1.0",
                "error": None,
            },
        ],
        "result_url": None,
        "error_message": None,
    }
    csv_output = processor_with_mock_predictor._generate_results_csv(
        cast(List[ResultItemDict], job_details["results"])
    )
    expected_csv = (
        "SMILES,BBB_Probability,Model_Version,Error\r\n"
        "CCO,0.8,1.0,\r\n"
        "invalid,,,Invalid SMILES\r\n"
        "CCC,0.3,1.0,\r\n"
    )
    assert csv_output == expected_csv
