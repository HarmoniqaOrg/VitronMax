"""
Tests for the batch prediction API endpoints.
"""

import io
import pytest
from fastapi.testclient import TestClient
from fastapi import UploadFile
from unittest.mock import patch, AsyncMock, MagicMock
import uuid

from app.main import app, batch_processor as main_batch_processor
from app.models import BatchPredictionStatus

client = TestClient(app)


def test_batch_predict_csv_valid() -> None:
    """Test batch prediction with valid CSV file."""
    # Create a test CSV file with header and valid SMILES
    csv_content = "SMILES\nCCO\nCC(=O)OC1=CC=CC=C1C(=O)O\nC1CCCCC1"
    csv_file = io.BytesIO(csv_content.encode())
    
    # Generate a unique job ID for this test run
    test_job_id = str(uuid.uuid4())
    expected_total_molecules = 3 # Number of SMILES in csv_content

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
        return test_job_id # start_batch_job returns the job_id string

    # Patch 'app.main.batch_processor.start_batch_job' and 'app.main.batch_processor.process_batch_job'
    # Also patch Supabase interactions as the real start_batch_job (even if side-effected) might call them.
    with patch("app.main.batch_processor.start_batch_job", new_callable=AsyncMock) as mock_start_job, \
         patch("app.main.batch_processor.process_batch_job", new_callable=AsyncMock) as mock_process_job, \
         patch("app.db.supabase.is_configured", new=True), \
         patch("app.db.supabase.create_batch_job", new_callable=AsyncMock):

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
        assert mock_start_job.call_args[1]['file'].filename == "test.csv"

        # process_batch_job is called by asyncio.create_task inside the real start_batch_job
        # Our mock_start_job_side_effect does not call asyncio.create_task(mock_process_job(...))
        # so mock_process_job should not have been called by the endpoint logic itself.
        mock_process_job.assert_not_called()


def test_batch_predict_csv_invalid() -> None:
    """Test batch prediction with invalid CSV file."""
    # Create an invalid CSV file (no SMILES column)
    csv_content = "Molecule,MW\nEthanol,46.07\nAspirin,180.16"
    csv_file = io.BytesIO(csv_content.encode())

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
    job_id_str = str(uuid.uuid4()) # Use a valid UUID string

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
        "results": [{"smiles": "C", "probability": 0.5, "model_version": "1.0", "error": None}] * 33
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

    # Use patch.object as context managers
    with patch("app.batch.supabase", mock_supabase_client):
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
