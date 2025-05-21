import pytest
from io import BytesIO
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from fastapi import UploadFile
from pytest_mock import MockerFixture

from app.models import BatchPredictionStatus
from app.batch import BatchProcessor
from app.main import batch_processor as main_batch_processor


def test_batch_predict_csv_valid(client: TestClient) -> None:
    """Test batch prediction with valid CSV file."""
    # Create a test CSV file with header and valid SMILES
    csv_content = "SMILES\nCCO\nCC(=O)OC1=CC=CC=C1C(=O)O\nC1CCCCC1"
    csv_file = BytesIO(csv_content.encode())

    # Generate a unique job ID for this test run
    test_job_id = str(uuid.uuid4())
    expected_total_molecules = 3  # Number of SMILES in csv_content

    # This function will be the side_effect of the mocked start_batch_job
    async def mock_start_job_side_effect(file: UploadFile) -> str:
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
            "error_message": None,
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
        assert (
            "file" in mock_start_job.call_args.kwargs
        ), "Mock was not called with 'file' keyword argument."
        uploaded_file_arg = mock_start_job.call_args.kwargs["file"]
        assert uploaded_file_arg.filename == "test.csv"
        mock_process_job.assert_not_called()


def test_batch_predict_csv_invalid(client: TestClient) -> None:
    """Test batch prediction with invalid CSV file."""
    # Create an invalid CSV file (no SMILES column)
    csv_content = "Molecule,MW\nEthanol,46.07\nAspirin,180.16"
    csv_file = BytesIO(csv_content.encode())

    with patch.object(main_batch_processor, "start_batch_job") as mock_start_job:
        mock_start_job.side_effect = ValueError(
            "Could not find SMILES column in CSV header"
        )

        response = client.post(
            "/batch_predict_csv", files={"file": ("invalid.csv", csv_file, "text/csv")}
        )

        assert response.status_code == 400
        assert "Could not find SMILES column" in response.json()["detail"]


def test_batch_status_valid(client: TestClient) -> None:
    """Test batch status endpoint with valid job ID."""
    job_id_str = str(uuid.uuid4())  # Use a valid UUID string

    # Ensure the job exists in active_jobs for the mocked batch_processor
    # The mock_global_batch_processor fixture ensures app.main.batch_processor is mocked
    main_batch_processor.active_jobs[job_id_str] = {
        "id": job_id_str,
        "status": BatchPredictionStatus.PROCESSING.value,
        "filename": "test.csv",
        "total_molecules": 10,
        "processed_molecules": 5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "results": [],
        "result_url": None,
        "error_message": None,
        "smiles_list": [],
    }

    response = client.get(f"/batch_predict_csv/{job_id_str}/status")

    assert response.status_code == 200, f"Response: {response.text}"
    data = response.json()
    assert data["id"] == job_id_str
    assert data["status"] == BatchPredictionStatus.PROCESSING.value
    assert data["progress"] == 50.0  # 5 out of 10
    assert data["filename"] == "test.csv"

    # Clean up the job from active_jobs to prevent interference with other tests
    if job_id_str in main_batch_processor.active_jobs:
        del main_batch_processor.active_jobs[job_id_str]


def test_batch_status_invalid(client: TestClient) -> None:
    """Test batch status endpoint with invalid job ID."""
    job_id = "nonexistent-job"

    # Patch get_job_status to raise ValueError for this specific test
    # The global batch_processor is already mocked by mock_global_batch_processor
    with patch.object(
        main_batch_processor, "get_job_status", side_effect=ValueError("Job not found")
    ) as mock_get_status:
        response = client.get(f"/batch_predict_csv/{job_id}/status")
        mock_get_status.assert_called_once_with(job_id)

    assert response.status_code == 404
    assert "Job not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_download_results_valid(
    client: TestClient,
    processor_with_mock_predictor: BatchProcessor,  # Fixture from conftest
    mocker: MockerFixture,
) -> None:
    """Test download endpoint with valid completed job."""
    job_id = "test-job-dl-valid-001"  # Unique ID
    processor_fixture_instance = (
        processor_with_mock_predictor  # Use the conftest fixture
    )

    processor_fixture_instance.active_jobs[job_id] = {
        "id": job_id,
        "status": BatchPredictionStatus.COMPLETED.value,
        "filename": "results_valid.csv",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_molecules": 2,
        "processed_molecules": 2,
        "smiles_list": ["CCO", "C1CCCCC1"],  # Needed if _generate_results_csv is called
        "results": [
            {
                "smiles": "CCO",
                "probability": 0.75,
                "model_version": "mock_v1.0",
                "error": None,
            },
            {
                "smiles": "C1CCCCC1",
                "probability": 0.85,
                "model_version": "mock_v1.0",
                "error": None,
            },
        ],
        "result_url": None,  # Ensure fallback to on-the-fly CSV generation
        "error_message": None,
    }

    with mocker.patch("app.main.batch_processor", new=processor_fixture_instance):
        mocker.patch.object(
            processor_fixture_instance,  # This is now app.main.batch_processor
            "get_job_status",
            return_value=processor_fixture_instance.active_jobs[job_id],
        )

        mock_csv_data = "SMILES,Probability,Model_Version,Error\nCCO,0.75,mock_v1.0,\nC1CCCCC1,0.85,mock_v1.0,"
        mocker.patch.object(
            processor_fixture_instance,
            "_generate_results_csv",
            return_value=mock_csv_data,
        )

        response = client.get(f"/batch_predict_csv/{job_id}/download")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/csv; charset=utf-8"
        assert (
            response.headers["content-disposition"]
            == 'attachment; filename="results_valid.csv"'
        )
        assert response.text == mock_csv_data

    if job_id in processor_fixture_instance.active_jobs:
        del processor_fixture_instance.active_jobs[job_id]


@pytest.mark.asyncio
async def test_download_results_not_completed(
    client: TestClient,
    processor_with_mock_predictor: BatchProcessor,  # Fixture from conftest
    mocker: MockerFixture,
) -> None:
    """Test download endpoint with job that is not yet completed."""
    job_id = "test-job-pending"
    processor_fixture_instance = processor_with_mock_predictor

    processor_fixture_instance.active_jobs[job_id] = {
        "id": job_id,
        "status": BatchPredictionStatus.PROCESSING.value,  # Not COMPLETED
        "filename": "pending_results.csv",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "total_molecules": 1,
        "processed_molecules": 0,
        "results": [],
        "result_url": None,
        "error_message": None,
        "smiles_list": [],
    }

    with mocker.patch("app.main.batch_processor", new=processor_fixture_instance):  # type: ignore[attr-defined]
        mocker.patch.object(
            processor_fixture_instance,
            "get_job_status",
            return_value=processor_fixture_instance.active_jobs[job_id],
        )
        response = client.get(f"/batch_predict_csv/{job_id}/download")

    assert response.status_code == 404  # Or 400 depending on implementation
    assert "Job not completed or results not available" in response.json()["detail"]

    if job_id in processor_fixture_instance.active_jobs:
        del processor_fixture_instance.active_jobs[job_id]


@pytest.mark.asyncio
async def test_supabase_storage_batch_results(
    client: TestClient,
    # processor_with_mock_predictor: BatchProcessor, # Not directly used, main_batch_processor is patched
    # mock_supabase_client_configured: MagicMock, # Not directly used
    mocker: MockerFixture,
) -> None:
    """
    Test that if Supabase is configured and results are uploaded,
    the download endpoint provides a signed URL.
    """
    job_id = str(uuid.uuid4())
    filename = "results_for_supabase.csv"
    mock_signed_url = (
        "https://supabase.example.com/signed/url/for/results_for_supabase.csv"
    )

    main_batch_processor.active_jobs[job_id] = {
        "id": job_id,
        "status": BatchPredictionStatus.COMPLETED.value,
        "filename": filename,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "total_molecules": 1,
        "processed_molecules": 1,
        "results": [
            {"smiles": "CCO", "probability": 0.7, "model_version": "v1", "error": None}
        ],
        "result_url": mock_signed_url,  # Crucially, result_url is already set
        "error_message": None,
        "smiles_list": ["CCO"],
    }

    mocker.patch.object(
        main_batch_processor,
        "get_job_status",
        return_value=main_batch_processor.active_jobs[job_id],
    )
    mocker.patch("app.db.supabase.is_configured", return_value=True)

    response = client.get(f"/batch_predict_csv/{job_id}/download")

    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Redirecting to signed URL for download."
    assert data["signed_url"] == mock_signed_url
    assert data["filename"] == filename

    spy_generate_csv = mocker.spy(main_batch_processor, "_generate_results_csv")
    spy_generate_csv.assert_not_called()
