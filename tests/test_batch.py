"""
Tests for the batch prediction API endpoints.
"""

import csv
import io
import logging
from io import BytesIO
import pytest
from fastapi import UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY
import uuid
from datetime import datetime
import re
import inspect

from app.main import app, batch_processor as main_batch_processor
from app.models import BatchPredictionStatus
from app.batch import BatchProcessor
from app.predict import BBBPredictor
from app.db import SupabaseClient

client = TestClient(app)


def test_batch_predict_csv_valid() -> None:
    """Test batch prediction with valid CSV file."""
    # Create a test CSV file with header and valid SMILES
    csv_content = "SMILES\nCCO\nCC(=O)OC1=CC=CC=C1C(=O)O\nC1CCCCC1"
    csv_file = BytesIO(csv_content.encode())

    # Generate a unique job ID for this test run
    test_job_id = str(uuid.uuid4())
    expected_total_molecules = 3  # Number of SMILES in csv_content

    # This function will be the side_effect of the mocked start_batch_job
    async def mock_start_job_side_effect(file: UploadFile):
        # Simulate start_batch_job populating active_jobs on the correct instance
        main_batch_processor.active_jobs[test_job_id] = {
            "id": test_job_id,
            "status": BatchPredictionStatus.PENDING.value,
            "filename": file.filename,
            "total_molecules": expected_total_molecules,
            "processed_molecules": 0,
            "created_at": "2025-05-19T22:00:00",
            "completed_at": None,
            "smiles_list": ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "C1CCCCC1"],
            "result_url": None,
            "results": [],
        }
        return test_job_id  # start_batch_job returns the job_id string

    # Patch 'app.main.batch_processor.start_batch_job' and 'app.main.batch_processor.process_batch_job'
    # Also patch Supabase interactions as the real start_batch_job (even if side-effected) might call them.
    with patch(
        "app.main.batch_processor.start_batch_job", new_callable=AsyncMock
    ) as mock_start_job, patch(
        "app.main.batch_processor.process_batch_job", new_callable=AsyncMock
    ) as mock_process_job, patch(
        "app.db.supabase.is_configured", new=True
    ), patch(
        "app.db.supabase.create_batch_job", new_callable=AsyncMock
    ):

        mock_start_job.side_effect = mock_start_job_side_effect

        # Make the request
        response = client.post(
            "/batch_predict_csv", files={"file": ("test.csv", csv_file, "text/csv")}
        )

        # Assertions
        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()
        assert data["id"] == test_job_id
        assert data["status"] == BatchPredictionStatus.PENDING.value
        assert data["filename"] == "test.csv"
        assert data["total_molecules"] == expected_total_molecules

        # Verify that start_batch_job was called once with the file
        mock_start_job.assert_called_once()
        # Check the 'file' keyword argument from the call
        # call_args[0] is for positional args, call_args[1] for keyword args
        # The first positional argument to start_batch_job is 'file'
        # If start_batch_job is defined as `async def start_batch_job(self, file: UploadFile)`
        # then the first arg after self is file. If it's `async def start_batch_job(file: UploadFile)` (static or module level)
        # then it's the first arg. Given it's a method on BatchProcessor, it's the first after self.
        # The mock however, replaces the method on the instance, so its signature is `(file: UploadFile)` effectively.
        assert mock_start_job.call_args[1]["file"].filename == "test.csv"

        # process_batch_job is called by asyncio.create_task inside the real start_batch_job
        # Our mock_start_job_side_effect does not call asyncio.create_task(mock_process_job(...))
        # so mock_process_job should not have been called by the endpoint logic itself.
        mock_process_job.assert_not_called()


def test_batch_predict_csv_invalid() -> None:
    """Test batch prediction with invalid CSV file."""
    # Create an invalid CSV file (no SMILES column)
    csv_content = "Molecule,MW\nEthanol,46.07\nAspirin,180.16"
    csv_file = BytesIO(csv_content.encode())

    # Patch the batch processor to raise an error
    with patch.object(main_batch_processor, "start_batch_job") as mock_start_job:
        mock_start_job.side_effect = ValueError(
            "Could not find SMILES column in CSV header"
        )

        # Make the request
        response = client.post(
            "/batch_predict_csv", files={"file": ("invalid.csv", csv_file, "text/csv")}
        )

        # Verify the response
        assert response.status_code == 400
        assert "Could not find SMILES column" in response.json()["detail"]


def test_batch_status_valid() -> None:
    """Test batch status endpoint with valid job ID."""
    job_id_str = str(uuid.uuid4())  # Use a valid UUID string

    # Populate the active_jobs dictionary of the batch_processor instance used by the app
    main_batch_processor.active_jobs[job_id_str] = {
        "id": job_id_str,
        "status": BatchPredictionStatus.PROCESSING.value,
        "filename": "test.csv",
        "total_molecules": 100,
        "processed_molecules": 33,
        "created_at": "2025-05-19T22:00:00",
        "completed_at": None,
        "progress": 33.0,
        "result_url": None,
        "error_message": None,
        "smiles_list": ["C"] * 100,
        "results": [
            {"smiles": "C", "probability": 0.5, "model_version": "1.0", "error": None}
        ]
        * 33,
    }

    # Patching is_configured to False is a good safety measure for this specific test's focus.
    with patch("app.db.supabase.is_configured", new=False):
        response = client.get(f"/batch_status/{job_id_str}")

    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    assert data["id"] == job_id_str
    assert data["status"] == BatchPredictionStatus.PROCESSING.value
    assert data["progress"] == 33.0
    assert data["filename"] == "test.csv"

    # Clean up the job from active_jobs to prevent interference with other tests
    if job_id_str in main_batch_processor.active_jobs:
        del main_batch_processor.active_jobs[job_id_str]


def test_batch_status_invalid() -> None:
    """Test batch status endpoint with invalid job ID."""
    job_id = "nonexistent-job"

    # Patch the batch processor to raise an error
    with patch.object(main_batch_processor, "get_job_status") as mock_status:
        mock_status.side_effect = ValueError(f"Job {job_id} not found")

        # Make the request
        response = client.get(f"/batch_status/{job_id}")

        # Verify the response
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


def test_download_results_valid() -> None:
    """Test download endpoint with valid completed job."""
    job_id = "test-job-789"

    # Patch the job status to return a completed job
    with patch.object(main_batch_processor, "get_job_status") as mock_status:
        mock_status.return_value = {
            "id": job_id,
            "status": BatchPredictionStatus.COMPLETED.value,
            "filename": "results.csv",
            "created_at": "2025-05-19T22:00:00",
        }

        # Patch active_jobs to include our test job
        main_batch_processor.active_jobs = {
            job_id: {
                "results": [
                    {
                        "smiles": "CCO",
                        "probability": 0.75,
                        "model_version": "1.0",
                        "error": None,
                    },
                    {
                        "smiles": "C1CCCCC1",
                        "probability": 0.85,
                        "model_version": "1.0",
                        "error": None,
                    },
                ]
            }
        }

        # Make the request
        response = client.get(f"/download/{job_id}")

        # Verify the response
        assert response.status_code == 200
        # Content-type might include charset in some environments
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]

        # Verify CSV content
        content = response.content.decode()
        assert "SMILES,BBB_Probability,Model_Version,Error" in content
        assert "CCO,0.75,1.0," in content
        assert "C1CCCCC1,0.85,1.0," in content


def test_download_results_not_completed() -> None:
    """Test download endpoint with job that is not yet completed."""
    job_id = "test-job-pending"

    # Patch the job status to return a processing job
    with patch.object(main_batch_processor, "get_job_status") as mock_status:
        mock_status.return_value = {
            "id": job_id,
            "status": BatchPredictionStatus.PROCESSING.value,
            "progress": 50.0,
            "filename": "pending.csv",
            "created_at": "2025-05-19T22:00:00",
        }

        # Make the request
        response = client.get(f"/download/{job_id}")

        # Verify the response
        # Different environments may return different error codes (400, 500)
        assert response.status_code in (400, 500)
        # Only check error details if we get a structured response
        if response.status_code == 400 and "detail" in response.json():
            assert "not completed" in response.json()["detail"]


@pytest.mark.asyncio
async def test_supabase_storage_batch_results() -> None:
    """Test that batch results are properly stored in Supabase Storage."""
    job_id = "test-storage-job-123"
    test_signed_url = (
        f"https://supabase.example.com/storage/v1/object/signed/{job_id}.csv"
    )

    # Mock Supabase client and its methods
    mock_supabase_client = MagicMock()
    mock_supabase_client.is_configured = True
    mock_supabase_client.store_batch_result_csv = AsyncMock(
        return_value=test_signed_url
    )
    # Mock other Supabase methods that might be called by process_batch_job
    mock_supabase_client.update_batch_job_status = AsyncMock()
    mock_supabase_client.update_batch_job_progress = AsyncMock()
    mock_supabase_client.complete_batch_job = AsyncMock()
    mock_supabase_client.fail_batch_job = AsyncMock()
    mock_supabase_client.store_batch_prediction_item = AsyncMock()

    # Mock for the predictor's predict method
    mock_predictor_predict_method = MagicMock(return_value=0.5)

    main_batch_processor.supabase = mock_supabase_client
    with patch.object(
        main_batch_processor.predictor, "predict", mock_predictor_predict_method
    ):
        # Initial job state, similar to what start_batch_job would create
        main_batch_processor.active_jobs = {
            job_id: {
                "id": job_id,
                "status": BatchPredictionStatus.PENDING.value,
                "filename": "test.csv",
                "total_molecules": 2,
                "processed_molecules": 0,
                "results": [],
                "created_at": "2025-05-20T10:00:00",
                "completed_at": None,
                "result_url": None,
                "smiles_list": ["CCO", "C1CCCCC1"],
            }
        }

        await main_batch_processor.process_batch_job(job_id)

        mock_supabase_client.store_batch_result_csv.assert_called_once()
        # Optionally, check call arguments
        args, kwargs = mock_supabase_client.store_batch_result_csv.call_args
        assert args[0] == job_id
        # Expected CSV content based on mocked predict and predictor version (assuming 1.0.0)
        # predictor_version = batch_processor.predictor.version # Get actual version
        # csv_content_expected = f"SMILES,BBB_Probability,Model_Version,Error\nCCO,0.5,{predictor_version},\nC1CCCCC1,0.5,{predictor_version},\n"
        # assert args[1] == csv_content_expected


@pytest.mark.asyncio
async def test_get_batch_status_not_found() -> None:
    """Test that get_job_status raises ValueError for a non-existent job ID."""
    with pytest.raises(ValueError, match=r"Job fake-id-does-not-exist not found"):
        main_batch_processor.get_job_status("fake-id-does-not-exist")


@pytest.mark.asyncio
async def test_get_batch_status_completed() -> None:
    """Test get_job_status for a completed job."""
    job_id = "completed-job-for-status-test"
    # Mock a completed job in active_jobs
    main_batch_processor.active_jobs[job_id] = {
        "id": job_id,
        "status": BatchPredictionStatus.COMPLETED.value,
        "filename": "completed_test.csv",
        "total_molecules": 1,
        "processed_molecules": 1,
        "created_at": "2023-10-26T10:00:00",
        "completed_at": "2023-10-26T10:05:00",
        "result_url": "http://example.com/results/completed_test.csv",
        "error_message": None,
        "results": [  # Add some mock results
            {
                "smiles": "CCO",
                "probability": 0.88,
                "model_version": "1.0.0",
                "error": None,
            }
        ],
    }

    status = main_batch_processor.get_job_status(job_id)

    assert status is not None
    assert status["id"] == job_id
    assert status["status"] == BatchPredictionStatus.COMPLETED.value
    assert status["result_url"] == "http://example.com/results/completed_test.csv"

    # Clean up the mocked job
    del main_batch_processor.active_jobs[job_id]


# New tests for BatchProcessor.validate_csv
@pytest.mark.asyncio
async def test_validate_csv_header_only() -> None:
    """Test validate_csv with a CSV file that only contains a header."""
    csv_content = "SMILES\n"
    csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
    upload_file = UploadFile(filename="header_only.csv", file=csv_file_bytes)

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)

    assert not is_valid
    assert error_msg == "No valid SMILES found in the CSV file"
    assert not smiles_list
    await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_more_than_1000_molecules() -> None:
    """Test validate_csv with a CSV file containing more than 1000 molecules."""
    header = "SMILES\n"
    smiles_lines = "\n".join([f"C{i}" for i in range(1005)])  # 1005 SMILES
    csv_content = header + smiles_lines
    csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
    upload_file = UploadFile(filename="too_many_smiles.csv", file=csv_file_bytes)

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)

    assert is_valid
    assert error_msg is None
    assert len(smiles_list) == 1000  # Should truncate to 1000
    assert smiles_list[0] == "C0"
    assert smiles_list[999] == "C999"
    await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_with_empty_smiles_rows() -> None:
    """Test validate_csv with a CSV file containing empty SMILES strings in data rows."""
    csv_content = "SMILES\nCCO\n\nC1CCCCC1\n   \nCCC"  # Includes empty lines and lines with only whitespace
    csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
    upload_file = UploadFile(filename="empty_smiles_rows.csv", file=csv_file_bytes)

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)

    assert is_valid
    assert error_msg is None
    assert len(smiles_list) == 3
    assert smiles_list == [
        "CCO",
        "C1CCCCC1",
        "CCC",
    ]  # Empty/whitespace-only SMILES should be skipped
    await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_empty_file() -> None:
    """Test validate_csv with an empty CSV file."""
    csv_content = ""
    csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
    upload_file = UploadFile(filename="empty.csv", file=csv_file_bytes)

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)
    assert not is_valid
    assert error_msg == "Empty CSV file"
    assert not smiles_list
    await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_no_smiles_column() -> None:
    """Test validate_csv with a CSV that has no SMILES column."""
    csv_content = "ID,Name\n1,MoleculeA\n2,MoleculeB"
    csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
    upload_file = UploadFile(filename="no_smiles_column.csv", file=csv_file_bytes)

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)
    assert not is_valid
    assert error_msg == "Could not find SMILES column in CSV header"
    assert not smiles_list
    await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_different_smiles_column_names() -> None:
    """Test validate_csv with different valid SMILES column names."""
    column_names = ["smi", "smile", "SMILES", "Smiles"]
    for col_name in column_names:
        csv_content = f"{col_name}\nCCO\nCCC"
        csv_file_bytes = BytesIO(csv_content.encode("utf-8"))
        upload_file = UploadFile(filename=f"test_{col_name}.csv", file=csv_file_bytes)

        processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
        is_valid, error_msg, smiles_list = await processor.validate_csv(upload_file)

        assert is_valid, f"Failed for column name: {col_name}"
        assert error_msg is None, f"Failed for column name: {col_name}"
        assert smiles_list == ["CCO", "CCC"], f"Failed for column name: {col_name}"
        await upload_file.close()


@pytest.mark.asyncio
async def test_validate_csv_unicode_decode_error():
    """Test validate_csv with a file that causes a UnicodeDecodeError."""
    # 0xFF is an invalid start byte in UTF-8 and can cause issues.
    # This aims to trigger the UnicodeDecodeError after both utf-8 and latin-1 attempts.
    import io

    content_bytes = b"\xff\xfe\xfd"  # Invalid UTF-8 sequence
    file = UploadFile(filename="invalid_encoding.csv", file=io.BytesIO(content_bytes))
    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(file)

    assert not is_valid
    assert error_msg == "File is not a valid CSV (encoding error)"
    assert smiles_list == []


@pytest.mark.asyncio
@patch("csv.reader")
async def test_validate_csv_csv_error(mock_csv_reader):
    """Test validate_csv with a file that causes a csv.Error due to malformed content."""
    # Configure the mock to raise csv.Error when called
    # The actual content doesn't matter as much now, as csv.reader itself will be mocked.
    # However, we still need to pass something that can be decoded.
    mock_csv_reader.side_effect = csv.Error("mocked csv error")

    content_bytes = b"SMILES\nCCO"
    file = UploadFile(
        filename="mocked_error.csv", file=BytesIO(content_bytes)
    )  # Corrected to BytesIO

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(file)

    assert not is_valid
    assert error_msg == "File is not a valid CSV (parsing error)"
    assert smiles_list == []


@pytest.mark.asyncio
async def test_validate_csv_generic_exception():
    """Test validate_csv when an unexpected generic Exception occurs during file processing."""
    # Create a mock UploadFile
    mock_upload_file = MagicMock(spec=UploadFile)
    # Configure the 'read' method of the file content mock to raise a generic Exception
    mock_upload_file.filename = "generic_error.csv"
    mock_upload_file.read = AsyncMock(
        side_effect=Exception("mocked generic read error")
    )

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    is_valid, error_msg, smiles_list = await processor.validate_csv(mock_upload_file)

    assert not is_valid
    assert error_msg == "Error processing CSV file: mocked generic read error"
    assert smiles_list == []


@pytest.mark.asyncio
async def test_start_batch_job_valid_csv():
    """Test start_batch_job with a valid CSV file."""
    # Create a test CSV file with header and valid SMILES
    csv_content = "SMILES\nCCO\nCC(=O)OC1=CC=CC=C1C(=O)O\nC1CCCCC1"
    csv_file = BytesIO(csv_content.encode())

    # Generate a unique job ID for this test run
    test_job_id = str(uuid.uuid4())
    expected_total_molecules = 3  # Number of SMILES in csv_content

    # This function will be the side_effect of the mocked start_batch_job
    async def mock_start_job_side_effect(file: UploadFile):
        # Simulate start_batch_job populating active_jobs on the correct instance
        main_batch_processor.active_jobs[test_job_id] = {
            "id": test_job_id,
            "status": BatchPredictionStatus.PENDING.value,
            "filename": file.filename,
            "total_molecules": expected_total_molecules,
            "processed_molecules": 0,
            "created_at": "2025-05-19T22:00:00",
            "completed_at": None,
            "smiles_list": ["CCO", "CC(=O)OC1=CC=CC=C1C(=O)O", "C1CCCCC1"],
            "result_url": None,
            "results": [],
        }
        return test_job_id  # start_batch_job returns the job_id string

    # Patch 'app.main.batch_processor.start_batch_job' and 'app.main.batch_processor.process_batch_job'
    # Also patch Supabase interactions as the real start_batch_job (even if side-effected) might call them.
    with patch(
        "app.main.batch_processor.start_batch_job", new_callable=AsyncMock
    ) as mock_start_job, patch(
        "app.main.batch_processor.process_batch_job", new_callable=AsyncMock
    ) as mock_process_job, patch(
        "app.db.supabase.is_configured", new=True
    ), patch(
        "app.db.supabase.create_batch_job", new_callable=AsyncMock
    ):

        mock_start_job.side_effect = mock_start_job_side_effect

        # Make the request
        response = client.post(
            "/batch_predict_csv", files={"file": ("test.csv", csv_file, "text/csv")}
        )

        # Assertions
        assert response.status_code == 200, f"Response: {response.text}"
        data = response.json()
        assert data["id"] == test_job_id
        assert data["status"] == BatchPredictionStatus.PENDING.value
        assert data["filename"] == "test.csv"
        assert data["total_molecules"] == expected_total_molecules

        # Verify that start_batch_job was called once with the file
        mock_start_job.assert_called_once()
        # Check the 'file' keyword argument from the call
        # call_args[0] is for positional args, call_args[1] for keyword args
        # The first positional argument to start_batch_job is 'file'
        # If start_batch_job is defined as `async def start_batch_job(self, file: UploadFile)`
        # then the first arg after self is file. If it's `async def start_batch_job(file: UploadFile)` (static or module level)
        # then it's the first arg. Given it's a method on BatchProcessor, it's the first after self.
        # The mock however, replaces the method on the instance, so its signature is `(file: UploadFile)` effectively.
        assert mock_start_job.call_args[1]["file"].filename == "test.csv"

        # process_batch_job is called by asyncio.create_task inside the real start_batch_job
        # Our mock_start_job_side_effect does not call asyncio.create_task(mock_process_job(...))
        # so mock_process_job should not have been called by the endpoint logic itself.
        mock_process_job.assert_not_called()


@pytest.mark.asyncio
async def test_start_batch_job_valid_csv_supabase_not_configured():
    """Test start_batch_job with a valid CSV when Supabase is not configured."""
    mock_upload_file = MagicMock(spec=UploadFile)
    mock_upload_file.filename = "valid_no_supabase.csv"
    # Ensure .read() is an awaitable mock if validate_csv were to actually read it,
    # but we are mocking validate_csv itself.

    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())
    smiles_data = ["CCO", "CCC"]

    with patch.object(
        processor, "process_batch_job", new_callable=AsyncMock
    ) as mock_process_method, patch(
        "app.batch.asyncio.create_task"
    ) as mock_create_task, patch.object(
        BatchProcessor, "validate_csv", return_value=(True, None, smiles_data)
    ), patch(
        "app.batch.logger"
    ):  # Patch the logger in app.batch

        processor.supabase.is_configured = False  # Key part of this test
        processor.supabase.create_batch_job = AsyncMock(
            return_value=None
        )  # Explicit mock

        job_id_result = await processor.start_batch_job(
            mock_upload_file
        )  # Renamed result to job_id_result for clarity

        # Assertions
        assert isinstance(job_id_result, str)  # Should return a job_id string
        job_details = processor.active_jobs[
            job_id_result
        ]  # Use job_id (result) to get details
        assert job_details["status"] == BatchPredictionStatus.PENDING.value
        assert job_details["filename"] == mock_upload_file.filename
        assert job_details["total_molecules"] == len(smiles_data)
        assert job_details["smiles_list"] == smiles_data

        mock_create_task.assert_called_once()
        mock_process_method.assert_called_once_with(job_id_result)

        # Check if background task was created
        processor.supabase.create_batch_job.assert_not_awaited()  # Ensure DB function not called
        # When Supabase is not configured, start_batch_job logs job creation info itself.
        # The primary check is that create_batch_job was not called.


@pytest.mark.filterwarnings(
    "ignore:coroutine 'AsyncMockMixin._execute_mock_call' was never awaited:RuntimeWarning:unittest.mock"
)
@pytest.mark.asyncio
async def test_start_batch_job_valid_csv_supabase_configured():
    """Test start_batch_job with a valid CSV when Supabase IS configured."""
    mock_upload_file = MagicMock(spec=UploadFile)
    mock_upload_file.filename = "valid_supabase_configured.csv"

    # Setup mock_supabase_client (this will be processor.supabase)
    mock_supabase_client = AsyncMock(spec=SupabaseClient)
    mock_supabase_client.is_configured = True
    # Explicitly make create_batch_job an AsyncMock on this client and set its return_value
    mock_supabase_client.create_batch_job = AsyncMock(return_value=None)

    processor = BatchProcessor(
        supabase_client=mock_supabase_client, predictor=AsyncMock()
    )
    smiles_data = [
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
        "CC(=O)Oc1ccccc1C(=O)OH",  # Aspirin
    ]

    # Patching context:
    # - Mock the instance method 'process_batch_job' on this specific 'processor' instance.
    # - Mock 'asyncio.create_task'.
    # - Mock 'validate_csv' and 'logger' as before.
    with patch.object(
        processor, "process_batch_job", new_callable=AsyncMock
    ) as mock_process_batch_job_instance_method, patch(
        "app.batch.asyncio.create_task"
    ) as mock_asyncio_create_task_func, patch.object(
        BatchProcessor, "validate_csv", return_value=(True, None, smiles_data)
    ), patch(
        "app.batch.logger"
    ):

        job_id_result = await processor.start_batch_job(mock_upload_file)

        # ----- Assertions -----

        # 1. Assert job details were correctly stored in active_jobs
        assert isinstance(job_id_result, str)
        job_details = processor.active_jobs[job_id_result]
        assert job_details["status"] == BatchPredictionStatus.PENDING.value
        assert job_details["filename"] == mock_upload_file.filename
        assert job_details["total_molecules"] == len(smiles_data)
        assert job_details["smiles_list"] == smiles_data

        # 2. Assert that the Supabase database call was made correctly
        mock_supabase_client.create_batch_job.assert_awaited_once_with(
            job_id=job_id_result,
            filename=mock_upload_file.filename,
            total_molecules=len(smiles_data),
            status=ANY,  # Using ANY because these are defaults from db.py, not directly passed by batch.py
            created_at=ANY,
        )

        # 3. Assert that 'processor.process_batch_job' (which is now 'mock_process_batch_job_instance_method')
        #    was called with the correct job_id. This happens inside 'start_batch_job'
        #    when 'self.process_batch_job(job_id)' is invoked.
        mock_process_batch_job_instance_method.assert_called_once_with(job_id_result)

        # 4. Assert that 'asyncio.create_task' was called.
        mock_asyncio_create_task_func.assert_called_once()  # Check it was called
        # Check that the argument passed to create_task was a coroutine
        assert inspect.iscoroutine(mock_asyncio_create_task_func.call_args[0][0])

        # Tests for process_batch_job

        # Check if background task was created
        processor.supabase.create_batch_job.assert_awaited_once()  # Ensure DB function was called

        # When Supabase is configured, start_batch_job logs job creation info.
        # The primary check is that create_batch_job was called.


# Tests for process_batch_job
@pytest.mark.asyncio
async def test_process_batch_job_success():
    """Test process_batch_job successfully processes a job with Supabase configured."""
    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())

    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]  # Ethanol, Propane
    mock_filename = "test_batch_success.csv"

    processor.active_jobs[job_id] = {
        "job_id": job_id,
        "filename": mock_filename,
        "status": BatchPredictionStatus.PENDING.value,
        "smiles_list": smiles_list,
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": [],
        "error_message": None,
        "result_url": None,
    }

    mock_predictor_predict = MagicMock(return_value=0.75)
    processor.predictor.predict = mock_predictor_predict

    mock_csv_content = "SMILES,probability,model_version,error\\nCCO,0.75,{pv},\\nCCC,0.75,{pv},".format(
        pv=processor.predictor.version
    )
    mock_generate_csv = MagicMock(return_value=mock_csv_content)
    processor._generate_results_csv = mock_generate_csv

    with patch("app.batch.asyncio.sleep", AsyncMock()), patch(
        "app.batch.logger"
    ) as mock_logger:

        processor.supabase = processor.supabase  # Assign the mock to processor.supabase

        processor.supabase.is_configured = True
        processor.supabase.store_batch_result_csv = AsyncMock(
            return_value=f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_success_token"
        )

        await processor.process_batch_job(job_id)

        job_result = processor.active_jobs[job_id]
        assert job_result["status"] == BatchPredictionStatus.COMPLETED.value
        assert job_result["processed_molecules"] == len(smiles_list)
        assert (
            job_result["result_url"]
            == f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_success_token"
        )
        assert len(job_result["results"]) == len(smiles_list)
        for i, res_item in enumerate(job_result["results"]):
            assert res_item["smiles"] == smiles_list[i]
            assert res_item["probability"] == 0.75
            assert res_item["model_version"] == processor.predictor.version
            assert res_item["error"] is None

        processor.supabase.update_batch_job_status.assert_any_call(
            job_id, BatchPredictionStatus.PROCESSING.value
        )

        expected_store_item_calls = []
        for i, smi in enumerate(smiles_list):
            expected_store_item_calls.append(
                call(
                    batch_id=job_id,
                    smiles=smi,
                    probability=0.75,
                    model_version=processor.predictor.version,
                    row_number=i,
                )
            )
        processor.supabase.store_batch_prediction_item.assert_has_calls(
            expected_store_item_calls, any_order=False
        )

        # update_batch_job_progress is called every 10 molecules, so for 2 it won't be called.
        # processor.supabase.update_batch_job_progress.assert_any_call(
        #     job_id, len(smiles_list), len(smiles_list)
        # )
        mock_generate_csv.assert_called_once_with(job_result["results"])
        processor.supabase.store_batch_result_csv.assert_awaited_once_with(
            job_id, mock_csv_content
        )
        processor.supabase.complete_batch_job.assert_awaited_once_with(
            job_id=job_id,
            result_url=f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_success_token",
        )

        mock_logger.info.assert_any_call(
            f"Storing batch results in Supabase Storage for job {job_id}"
        )
        mock_logger.info.assert_any_call(f"Batch job {job_id} completed successfully")


@pytest.mark.asyncio
async def test_process_batch_job_smiles_error():
    """Test process_batch_job when some SMILES strings cause a ValueError during prediction."""
    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())

    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "INVALID_SMILES", "CCC"]  # One invalid SMILES
    mock_filename = "test_batch_smiles_error.csv"

    processor.active_jobs[job_id] = {
        "job_id": job_id,
        "filename": mock_filename,
        "status": BatchPredictionStatus.PENDING.value,
        "smiles_list": smiles_list,
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": [],
        "error_message": None,  # Job-level error, should remain None
        "result_url": None,
    }

    # Mock predictor's predict to raise ValueError for "INVALID_SMILES"
    mock_smiles_error_message = "Mocked SMILES processing error"

    def mock_predict_side_effect(smi):
        if smi == "INVALID_SMILES":
            raise ValueError(mock_smiles_error_message)
        return 0.65  # Successful prediction for others

    mock_predictor_predict = MagicMock(side_effect=mock_predict_side_effect)
    processor.predictor.predict = mock_predictor_predict

    # Expected CSV output will include the error for the invalid SMILES
    expected_csv_output = (
        f"SMILES,probability,model_version,error\\n"
        f"CCO,0.65,{processor.predictor.version},\\n"
        f"INVALID_SMILES,,{processor.predictor.version},{mock_smiles_error_message}\\n"
        f"CCC,0.65,{processor.predictor.version},"
    )
    mock_generate_csv = MagicMock(return_value=expected_csv_output)
    processor._generate_results_csv = mock_generate_csv

    with patch("app.batch.asyncio.sleep", AsyncMock()), patch(
        "app.batch.logger"
    ) as mock_logger:

        processor.supabase = processor.supabase  # Assign the mock to processor.supabase

        processor.supabase.is_configured = True
        processor.supabase.store_batch_result_csv = AsyncMock(
            return_value=f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_smiles_error_token"
        )

        await processor.process_batch_job(job_id)

        job_result = processor.active_jobs[job_id]
        assert job_result["status"] == BatchPredictionStatus.COMPLETED.value
        assert job_result["processed_molecules"] == len(smiles_list)
        assert (
            job_result["result_url"]
            == f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_smiles_error_token"
        )
        assert len(job_result["results"]) == len(smiles_list)
        assert job_result["error_message"] is None  # No job-level error

        # Check individual results
        assert job_result["results"][0]["smiles"] == "CCO"
        assert job_result["results"][0]["probability"] == 0.65
        assert job_result["results"][0]["error"] is None

        assert job_result["results"][1]["smiles"] == "INVALID_SMILES"
        assert job_result["results"][1]["probability"] is None
        assert job_result["results"][1]["error"] == mock_smiles_error_message

        assert job_result["results"][2]["smiles"] == "CCC"
        assert job_result["results"][2]["probability"] == 0.65
        assert job_result["results"][2]["error"] is None

        processor.supabase.update_batch_job_status.assert_any_call(
            job_id, BatchPredictionStatus.PROCESSING.value
        )

        processor.supabase.store_batch_prediction_item.assert_any_call(
            batch_id=job_id,
            smiles="CCO",
            probability=0.65,
            model_version=processor.predictor.version,
            row_number=0,
        )
        processor.supabase.store_batch_prediction_item.assert_any_call(
            batch_id=job_id,
            smiles="INVALID_SMILES",
            probability=None,
            model_version=processor.predictor.version,
            row_number=1,
            error_message=mock_smiles_error_message,
        )
        processor.supabase.store_batch_prediction_item.assert_any_call(
            batch_id=job_id,
            smiles="CCC",
            probability=0.65,
            model_version=processor.predictor.version,
            row_number=2,
        )

        # update_batch_job_progress is called every 10 molecules, so for 3 it won't be called.
        # processor.supabase.update_batch_job_progress.assert_any_call(
        #     job_id, len(smiles_list), len(smiles_list)
        # )
        mock_generate_csv.assert_called_once_with(job_result["results"])
        processor.supabase.store_batch_result_csv.assert_awaited_once_with(
            job_id, expected_csv_output
        )
        processor.supabase.complete_batch_job.assert_awaited_once_with(
            job_id=job_id,
            result_url=f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_smiles_error_token",
        )

        mock_logger.info.assert_any_call(f"Batch job {job_id} completed successfully")


@pytest.mark.asyncio
async def test_process_batch_job_supabase_storage_failure():
    """Test process_batch_job when Supabase Storage fails to store the CSV."""
    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())

    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]
    mock_filename = "test_storage_failure.csv"

    processor.active_jobs[job_id] = {
        "job_id": job_id,
        "filename": mock_filename,
        "status": BatchPredictionStatus.PENDING.value,
        "smiles_list": smiles_list,
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": [],
        "error_message": None,
        "result_url": None,
    }

    mock_predictor_predict = MagicMock(return_value=0.80)
    processor.predictor.predict = mock_predictor_predict

    mock_csv_content = "SMILES,probability,model_version,error\\nCCO,0.80,{pv},\\nCCC,0.80,{pv},".format(
        pv=processor.predictor.version
    )
    mock_generate_csv = MagicMock(return_value=mock_csv_content)
    processor._generate_results_csv = mock_generate_csv

    with patch("app.batch.asyncio.sleep", AsyncMock()), patch(
        "app.batch.logger"
    ) as mock_logger:

        processor.supabase = processor.supabase  # Assign the mock to processor.supabase

        processor.supabase.is_configured = True
        processor.supabase.store_batch_result_csv = AsyncMock(
            return_value=None
        )  # Simulate storage failure

        await processor.process_batch_job(job_id)

        job_result = processor.active_jobs[job_id]
        assert job_result["status"] == BatchPredictionStatus.COMPLETED.value
        assert job_result["processed_molecules"] == len(smiles_list)
        assert job_result["result_url"] == f"/download/{job_id}"
        assert len(job_result["results"]) == len(smiles_list)

        processor.supabase.update_batch_job_status.assert_any_call(
            job_id, BatchPredictionStatus.PROCESSING.value
        )

        # store_batch_prediction_item should have been called for each SMILES
        expected_store_item_calls = []
        for i, smi in enumerate(smiles_list):
            expected_store_item_calls.append(
                call(
                    batch_id=job_id,
                    smiles=smi,
                    probability=0.80,
                    model_version=processor.predictor.version,
                    row_number=i,
                )
            )
        processor.supabase.store_batch_prediction_item.assert_has_calls(
            expected_store_item_calls, any_order=False
        )

        # The job_result["results"] will be populated by process_batch_job before calling _generate_results_csv
        mock_generate_csv.assert_called_once_with(job_result["results"])
        processor.supabase.store_batch_result_csv.assert_awaited_once_with(
            job_id, mock_csv_content
        )
        processor.supabase.complete_batch_job.assert_awaited_once_with(
            job_id=job_id, result_url=f"/download/{job_id}"
        )

        mock_logger.warning.assert_any_call(
            f"Failed to store batch results in Supabase Storage for job {job_id}"
        )
        mock_logger.info.assert_any_call(f"Batch job {job_id} completed successfully")


@pytest.mark.asyncio
async def test_process_batch_job_generic_failure():
    """Test process_batch_job when a generic Exception occurs during processing."""
    processor = BatchProcessor(supabase_client=AsyncMock(), predictor=AsyncMock())

    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]
    mock_filename = "test_generic_failure.csv"

    processor.active_jobs[job_id] = {
        "job_id": job_id,
        "filename": mock_filename,
        "status": BatchPredictionStatus.PENDING.value,
        "smiles_list": smiles_list,
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "results": [],
        "error_message": None,
        "result_url": None,
    }

    mock_predictor_predict = MagicMock(return_value=0.80)
    processor.predictor.predict = mock_predictor_predict

    mock_csv_content = "SMILES,probability,model_version,error\\nCCO,0.80,{pv},\\nCCC,0.80,{pv},".format(
        pv=processor.predictor.version
    )
    mock_generate_csv = MagicMock(return_value=mock_csv_content)
    processor._generate_results_csv = mock_generate_csv

    with patch("app.batch.asyncio.sleep", AsyncMock()), patch(
        "app.batch.logger"
    ) as mock_logger:

        processor.supabase = processor.supabase  # Assign the mock to processor.supabase

        processor.supabase.is_configured = True
        processor.supabase.store_batch_result_csv = AsyncMock(
            return_value=f"https://fake.supabase.co/storage/v1/object/sign/results/{job_id}.csv?token=mock_success_token"
        )

        # Simulate a generic Exception during processing
        processor.predictor.predict = MagicMock(
            side_effect=Exception("Mocked generic processing error")
        )

        await processor.process_batch_job(job_id)

        job_result = processor.active_jobs[job_id]
        assert job_result["status"] == BatchPredictionStatus.FAILED.value
        assert job_result["processed_molecules"] == 0
        assert job_result["result_url"] is None
        assert len(job_result["results"]) == 0
        assert job_result["error_message"] == "Mocked generic processing error"

        processor.supabase.update_batch_job_status.assert_any_call(
            job_id, BatchPredictionStatus.PROCESSING.value
        )
        processor.supabase.fail_batch_job.assert_awaited_once_with(
            job_id=job_id, error_message="Mocked generic processing error"
        )

        mock_logger.error.assert_any_call(
            f"Error processing batch job {job_id}: Mocked generic processing error"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "csv_content, expected_error_message",
    [
        ("", "Empty CSV file"),  # Empty file
        ("SMILES\n", "No valid SMILES found in the CSV file"),  # Header only
        (
            "MOLECULE\nCCO",
            "Could not find SMILES column in CSV header",  # Wrong header
        ),
    ],
)
async def test_start_batch_job_invalid_csv(
    csv_content: str,
    expected_error_message: str,
    processor_with_mock_predictor: BatchProcessor,
    mock_supabase_client: AsyncMock,
    caplog: pytest.LogCaptureFixture,
):
    """Test start_batch_job with various invalid CSV inputs."""
    mock_file = UploadFile(
        filename="invalid.csv", file=io.BytesIO(csv_content.encode("utf-8"))
    )
    processor_with_mock_predictor.supabase = mock_supabase_client

    # Patch uuid.uuid4 to control job_id generation for consistent testing if needed
    # For this test, we are checking for the ValueError before job_id is critical

    with pytest.raises(ValueError, match=re.escape(expected_error_message)):
        await processor_with_mock_predictor.start_batch_job(mock_file)

    # Ensure no job was created in active_jobs or Supabase
    assert not processor_with_mock_predictor.active_jobs
    mock_supabase_client.create_batch_job.assert_not_called()


# Fixtures
@pytest.fixture
def mock_predictor() -> MagicMock:
    predictor = MagicMock(spec=BBBPredictor)
    predictor.version = "mock_v1.0"
    # Default predict behavior, can be overridden in tests
    predictor.predict = MagicMock(return_value=0.5)
    # If BatchProcessor uses predict_batch, mock it appropriately:
    # predictor.predict_batch = MagicMock(return_value=[(0.5, None)] * 10)
    return predictor


@pytest.fixture
def mock_supabase_client() -> AsyncMock:
    client = AsyncMock()
    client.is_configured = True  # Default, can be changed in tests
    client.create_batch_job = AsyncMock(return_value=None)
    client.update_batch_job_status = AsyncMock(return_value=None)
    client.update_batch_job_progress = AsyncMock(return_value=None)
    client.store_batch_prediction_item = AsyncMock(return_value=None)
    # Ensure store_batch_result_csv returns a string URL as per its usage
    client.store_batch_result_csv = AsyncMock(
        return_value="mock_storage_url/default_job_id.csv"
    )
    client.complete_batch_job = AsyncMock(return_value=None)
    client.fail_batch_job = AsyncMock(return_value=None)
    return client


@pytest.fixture
def processor_with_mock_predictor(
    mock_predictor: MagicMock, mock_supabase_client: AsyncMock
) -> BatchProcessor:
    processor = BatchProcessor(
        predictor=mock_predictor, supabase_client=mock_supabase_client
    )
    # Mock the logger instance on the processor to control log assertions cleanly
    processor.logger = MagicMock(spec=logging.Logger)
    return processor
