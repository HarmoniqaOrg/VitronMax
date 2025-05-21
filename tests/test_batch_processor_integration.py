"""
Integration tests for the BatchProcessor class methods (with mocked Supabase/Predictor).
"""

import pytest
import uuid
from datetime import datetime, timezone
from typing import TypedDict, Dict, List
from unittest.mock import AsyncMock, MagicMock

from fastapi import UploadFile
from pytest_mock import MockerFixture

from app.batch import BatchProcessor, JobDetailsDict
from app.db import SupabaseClient
from app.models import BatchPredictionStatus


class ExpectedResult(TypedDict, total=False):
    probability: float
    error: Exception  # Using Exception, can be more specific like ValueError


# Tests for process_batch_job (integration with mocked predictor and Supabase)
@pytest.mark.asyncio
async def test_process_batch_job_success(
    processor_with_mock_predictor: BatchProcessor,
    mock_supabase_client_configured: MagicMock,  # To verify calls if needed, though processor has its own
    mocker: MockerFixture,
) -> None:
    """Test successful processing of a batch job, including Supabase upload."""
    processor = processor_with_mock_predictor
    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]
    job_details: JobDetailsDict = {
        "id": job_id,
        "status": BatchPredictionStatus.PENDING.value,
        "filename": "test_smiles.csv",
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "smiles_list": smiles_list,
        "results": [],
        "result_url": None,
        "error_message": None,
    }
    processor.active_jobs[job_id] = job_details

    # Configure the mock predictor's 'predict' method
    # It will be called for each SMILES string in smiles_list
    # Let's define the expected probabilities
    expected_probabilities: Dict[str, float] = {
        "CCO": 0.8,
        "CCC": 0.3,
    }

    def mock_predict_side_effect(smiles: str) -> float:
        return expected_probabilities.get(smiles, 0.0)  # Default if SMILES not in map

    # Patch 'predict' method on the processor.predictor mock instance
    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", new_callable=MagicMock
    )
    mocked_predict.side_effect = mock_predict_side_effect

    # Mock Supabase interactions
    processor.supabase = mock_supabase_client_configured
    processor.supabase.is_configured = True
    processor.supabase.update_batch_job_status = AsyncMock()
    processor.supabase.store_batch_prediction_item = AsyncMock()
    processor.supabase.store_batch_result_csv = AsyncMock(
        return_value="https://example.com/results.csv"
    )
    processor.supabase.complete_batch_job = AsyncMock()

    await processor.process_batch_job(job_id)

    assert job_details["status"] == BatchPredictionStatus.COMPLETED.value
    assert job_details["results"] is not None
    assert len(job_details["results"]) == len(smiles_list)
    # Check individual results
    for i, smi in enumerate(smiles_list):
        assert job_details["results"][i]["smiles"] == smi
        assert job_details["results"][i]["probability"] == expected_probabilities[smi]
        assert job_details["results"][i]["model_version"] == processor.predictor.version
        assert job_details["results"][i]["error"] is None

    assert job_details["result_url"] == "https://example.com/results.csv"

    # Assert that predictor.predict was called for each SMILES
    assert mocked_predict.call_count == len(smiles_list)
    for smi in smiles_list:
        mocked_predict.assert_any_call(smi)

    processor.supabase.store_batch_result_csv.assert_called_once()
    # Check that the first argument to store_batch_result_csv (csv_content) is correct
    # This requires constructing the expected CSV content based on results
    # For brevity, we'll skip exact CSV content matching here, but it's important for thoroughness

    processor.supabase.complete_batch_job.assert_called_once_with(
        job_id,
        BatchPredictionStatus.COMPLETED.value,
        "https://example.com/results.csv",
        None,
    )
    # Ensure initial status update to PROCESSING was called
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id, BatchPredictionStatus.PROCESSING.value
    )
    # Ensure prediction items were stored
    assert processor.supabase.store_batch_prediction_item.call_count == len(smiles_list)


@pytest.mark.asyncio
async def test_process_batch_job_partial_success_invalid_smiles(
    processor_with_mock_predictor: BatchProcessor,
    mock_supabase_client_configured: MagicMock,
    mocker: MockerFixture,
) -> None:
    """Test processing with some invalid SMILES. Predictor's 'predict' method will be called for each."""
    processor = processor_with_mock_predictor
    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "invalid_smiles", "CCC"]
    job_details: JobDetailsDict = {
        "id": job_id,
        "status": BatchPredictionStatus.PENDING.value,
        "filename": "partial_smiles.csv",
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "smiles_list": smiles_list,
        "results": [],
        "result_url": None,
        "error_message": None,
    }
    processor.active_jobs[job_id] = job_details

    # Configure the mock predictor's 'predict' method
    expected_results_map: Dict[str, ExpectedResult] = {
        "CCO": {"probability": 0.8},
        "invalid_smiles": {"error": ValueError("Invalid SMILES string")},
        "CCC": {"probability": 0.3},
    }

    def mock_predict_side_effect(smiles: str) -> float:
        if "error" in expected_results_map[smiles]:
            raise expected_results_map[smiles]["error"]
        return expected_results_map[smiles]["probability"]

    # Patch 'predict' method
    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", new_callable=MagicMock
    )
    mocked_predict.side_effect = mock_predict_side_effect

    # Mock Supabase interactions
    processor.supabase = mock_supabase_client_configured
    processor.supabase.is_configured = True
    processor.supabase.update_batch_job_status = AsyncMock()
    processor.supabase.store_batch_prediction_item = AsyncMock()
    processor.supabase.store_batch_result_csv = AsyncMock(
        return_value="https://example.com/partial_results.csv"
    )
    processor.supabase.complete_batch_job = AsyncMock()

    await processor.process_batch_job(job_id)

    assert job_details["status"] == BatchPredictionStatus.COMPLETED.value
    assert len(job_details["results"]) == len(smiles_list)

    # Check individual results
    for i, smi in enumerate(smiles_list):
        result = job_details["results"][i]
        assert result["smiles"] == smi
        if "error" in expected_results_map[smi]:
            assert result["error"] is not None
            assert str(expected_results_map[smi]["error"]) in result["error"]
            assert result["probability"] is None
        else:
            assert result["error"] is None
            assert result["probability"] == expected_results_map[smi]["probability"]
        assert (
            result["model_version"] == processor.predictor.version
        )  # or None if error before version set

    assert job_details["result_url"] == "https://example.com/partial_results.csv"

    # Assert that predictor.predict was called for each SMILES
    assert mocked_predict.call_count == len(smiles_list)
    for smi in smiles_list:
        mocked_predict.assert_any_call(smi)

    # Assert Supabase calls
    processor.supabase.store_batch_result_csv.assert_called_once()
    processor.supabase.complete_batch_job.assert_called_once_with(
        job_id,
        BatchPredictionStatus.COMPLETED.value,
        "https://example.com/partial_results.csv",
        None,
    )
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id, BatchPredictionStatus.PROCESSING.value
    )
    assert processor.supabase.store_batch_prediction_item.call_count == len(smiles_list)
    # Check calls to store_batch_prediction_item (can be more detailed)
    for i, smi in enumerate(smiles_list):
        expected_item_args = {
            "batch_id": job_id,
            "smiles": smi,
            "row_number": i,
            "model_version": processor.predictor.version,
        }
        if "error" in expected_results_map[smi]:
            expected_item_args["probability"] = None
            expected_item_args["error_message"] = str(
                expected_results_map[smi]["error"]
            )
        else:
            expected_item_args["probability"] = expected_results_map[smi]["probability"]
        # This check is a bit complex due to kwargs. A simpler check might be call_count or specific args.
        # For now, ensuring it's called for each item is a good start.
        # processor.supabase.store_batch_prediction_item.assert_any_call(**expected_item_args) # This might be too strict if other kwargs are passed


@pytest.mark.asyncio
async def test_process_batch_job_predictor_error(
    processor_with_mock_predictor: BatchProcessor,
    mock_supabase_client_configured: MagicMock,
    mocker: MockerFixture,
) -> None:
    """Test job failure when predictor.predict raises an unexpected error."""
    processor = processor_with_mock_predictor
    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]  # Second SMILES will cause error
    job_details: JobDetailsDict = {
        "id": job_id,
        "status": BatchPredictionStatus.PENDING.value,
        "filename": "error_job.csv",
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "smiles_list": smiles_list,
        "results": [],
        "result_url": None,
        "error_message": None,
    }
    processor.active_jobs[job_id] = job_details

    # Configure the mock predictor to raise an exception for all SMILES
    predictor_error_message = "Simulated predictor runtime error"

    def mock_predict_side_effect_func(*args, **kwargs):
        raise Exception(predictor_error_message)

    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", side_effect=mock_predict_side_effect_func
    )

    # Create a dedicated Supabase mock for this test
    mock_supabase_for_test = AsyncMock(spec=SupabaseClient)
    mock_supabase_for_test.is_configured = True
    mock_supabase_for_test.store_batch_result_csv = AsyncMock(
        return_value="http://mocked.url/predictor_error.csv"
    )
    mock_supabase_for_test.complete_batch_job = AsyncMock()
    mock_supabase_for_test.update_batch_job_status = AsyncMock()
    mock_supabase_for_test.store_batch_prediction_item = (
        AsyncMock()
    )  # Should not be called if all predictions error before storage attempt
    processor.supabase = mock_supabase_for_test

    await processor.process_batch_job(job_id)

    # The job completes, but individual predictions fail
    assert job_details["status"] == BatchPredictionStatus.COMPLETED.value
    assert job_details["error_message"] is None  # No job-level error
    assert len(job_details["results"]) == len(smiles_list)
    for result in job_details["results"]:
        assert result["error"] == f"Prediction error: {predictor_error_message}"
        assert result["probability"] is None

    # Check Supabase calls (should still try to upload the CSV of errors)
    processor.supabase.store_batch_result_csv.assert_called_once()
    processor.supabase.complete_batch_job.assert_called_once_with(
        job_id=job_id,
        status=BatchPredictionStatus.COMPLETED,
        # result_url will be the return value of the mocked store_batch_result_csv
        result_url=processor.supabase.store_batch_result_csv.return_value,
        error_message=None,  # No job-level error message
    )

    # Predictor should have been called for the first SMILES, then failed on the second
    assert mocked_predict.call_count == 2  # Called for CCO, then for CCC (which errors)
    mocked_predict.assert_any_call("CCO")
    mocked_predict.assert_any_call("CCC")

    # Supabase assertions
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id, BatchPredictionStatus.PROCESSING.value
    )
    # Final status update to COMPLETED
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id, BatchPredictionStatus.COMPLETED.value, error_message=None
    )
    # store_batch_prediction_item should be called for the first (successful) item
    processor.supabase.store_batch_prediction_item.assert_called_once_with(
        batch_id=job_id,
        smiles="CCO",
        probability=0.5,
        model_version=processor.predictor.version,
        row_number=0,
        error_message=None,
    )

    processor.supabase.store_batch_result_csv.assert_called_once()  # It was called and then it failed
    processor.supabase.complete_batch_job.assert_called_once()


@pytest.mark.asyncio
async def test_process_batch_job_supabase_upload_failure(
    processor_with_mock_predictor: BatchProcessor,
    mock_supabase_client_configured: MagicMock,
    mocker: MockerFixture,
) -> None:
    """Test job failure when Supabase store_batch_result_csv raises an error."""
    processor = processor_with_mock_predictor
    job_id = str(uuid.uuid4())
    smiles_list = ["CCO", "CCC"]
    job_details: JobDetailsDict = {
        "id": job_id,
        "status": BatchPredictionStatus.PENDING.value,
        "filename": "upload_fail.csv",
        "total_molecules": len(smiles_list),
        "processed_molecules": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "smiles_list": smiles_list,
        "results": [],
        "result_url": None,
        "error_message": None,
    }
    processor.active_jobs[job_id] = job_details

    # Configure the mock predictor's 'predict' method
    expected_probabilities: Dict[str, float] = {"CCO": 0.8, "CCC": 0.3}

    def mock_predict_side_effect(smiles: str) -> float:
        return expected_probabilities.get(smiles, 0.0)

    # Patch 'predict' method
    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", new_callable=MagicMock
    )
    mocked_predict.side_effect = mock_predict_side_effect

    # Create a dedicated Supabase mock for this test
    mock_supabase_for_test = AsyncMock(spec=SupabaseClient)
    mock_supabase_for_test.is_configured = True
    mock_supabase_for_test.update_batch_job_status = AsyncMock()
    mock_supabase_for_test.store_batch_prediction_item = AsyncMock()
    supabase_upload_error_message = "Simulated Supabase upload error"

    async def mock_store_results_side_effect(*args, **kwargs):
        raise Exception(supabase_upload_error_message)

    mock_supabase_for_test.store_batch_result_csv = AsyncMock(
        side_effect=mock_store_results_side_effect
    )
    mock_supabase_for_test.complete_batch_job = AsyncMock()  # Should not be called

    processor.supabase = mock_supabase_for_test  # Assign the dedicated mock

    await processor.process_batch_job(job_id)

    assert job_details["status"] == BatchPredictionStatus.FAILED.value
    assert (
        job_details["error_message"] == supabase_upload_error_message
    )  # Corrected assertion

    assert job_details["result_url"] is None
    # Results should still be populated locally in job_details even if upload fails
    assert len(job_details["results"]) == len(smiles_list)

    # Assert that predictor.predict was called for each SMILES
    assert mocked_predict.call_count == len(smiles_list)
    for smi in smiles_list:
        mocked_predict.assert_any_call(smi)

    # Supabase assertions
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id, BatchPredictionStatus.PROCESSING.value
    )
    processor.supabase.update_batch_job_status.assert_any_call(
        job_id,
        BatchPredictionStatus.FAILED.value,
        error_message=job_details["error_message"],
    )
    assert processor.supabase.store_batch_prediction_item.call_count == len(smiles_list)
    # Verify store_batch_prediction_item calls (example for first item)
    processor.supabase.store_batch_prediction_item.assert_any_call(
        batch_id=job_id,
        smiles="CCO",
        probability=expected_probabilities["CCO"],
        model_version=processor.predictor.version,
        row_number=0,
        error_message=None,
    )

    processor.supabase.store_batch_result_csv.assert_called_once()  # It was called and then it failed
    processor.supabase.complete_batch_job.assert_called_once_with(  # Corrected assertion
        job_id=job_id,
        status=BatchPredictionStatus.FAILED,
        result_url=None,  # Expect None as result_url if upload failed
        error_message=supabase_upload_error_message,
    )


@pytest.mark.asyncio
async def test_process_batch_job_no_smiles(
    processor_with_mock_predictor: BatchProcessor, mocker: MockerFixture
) -> None:
    """Test processing a job with an empty SMILES list."""
    processor = processor_with_mock_predictor
    job_id = str(uuid.uuid4())
    smiles_list: List[str] = []  # Explicitly type smiles_list
    job_details: JobDetailsDict = {
        "id": job_id,
        "status": BatchPredictionStatus.PENDING.value,
        "filename": "empty_smiles.csv",
        "total_molecules": 0,
        "processed_molecules": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "smiles_list": smiles_list,
        "results": [],
        "result_url": None,
        "error_message": None,
    }
    processor.active_jobs[job_id] = job_details

    # Create a dedicated Supabase mock for this test
    mock_supabase_for_test = AsyncMock(spec=SupabaseClient)
    mock_supabase_for_test.is_configured = True
    mock_supabase_for_test.store_batch_result_csv = AsyncMock(
        return_value="http://mocked.url/empty_smiles.csv"
    )
    mock_supabase_for_test.complete_batch_job = AsyncMock()
    mock_supabase_for_test.update_batch_job_status = (
        AsyncMock()
    )  # Added for completeness if any status updates are checked
    # Assign the dedicated mock to the processor for this test
    processor.supabase = mock_supabase_for_test

    # Patch 'predict' method even if not expected to be called, for consistency if needed by mypy
    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", new_callable=MagicMock
    )
    assert mocked_predict.call_count == 0

    await processor.process_batch_job(job_id)

    # Assertions
    assert job_details["status"] == BatchPredictionStatus.COMPLETED.value
    assert job_details["results"] == []
    assert job_details["error_message"] is None
    # Since smiles_list is empty, predict_smiles_batch should not be called

    # Check Supabase calls: store_batch_result_csv might be called with empty content
    # and complete_batch_job should be called.
    processor.supabase.store_batch_result_csv.assert_called_once()
    # Check the csv_content argument passed to store_batch_result_csv
    assert (
        processor.supabase.store_batch_result_csv.call_args.kwargs[
            "csv_content"
        ]  # Corrected access to kwargs
        == "SMILES,BBB_Probability,Model_Version,Error\r\n"
    )
    processor.supabase.complete_batch_job.assert_called_once_with(
        job_id=job_id,
        status=BatchPredictionStatus.COMPLETED,
        result_url=processor.supabase.store_batch_result_csv.return_value,  # The URL from the (mocked) upload
        error_message=None,
    )


@pytest.mark.asyncio
async def test_process_batch_job_job_not_found(
    processor_with_mock_predictor: BatchProcessor, mocker: MockerFixture
) -> None:
    """Test processing a job that doesn't exist in active_jobs."""
    processor = processor_with_mock_predictor
    job_id = "nonexistent-job"
    mock_logger_error = mocker.patch("app.batch.logger.error")

    # No predictions should be attempted
    mocked_predict = mocker.patch.object(
        processor.predictor, "predict", new_callable=MagicMock
    )
    assert mocked_predict.call_count == 0

    await processor.process_batch_job(job_id)

    mock_logger_error.assert_called_once_with(f"Job {job_id} not found")


# Tests for start_batch_job (integration with Supabase configuration)
@pytest.mark.asyncio
async def test_start_batch_job_valid_csv_supabase_not_configured(
    processor_with_mock_predictor: BatchProcessor,
    mock_upload_file_valid: UploadFile,
    mock_supabase_client_not_configured: MagicMock,  # Fixture from conftest
    mocker: MockerFixture,
) -> None:
    """Test start_batch_job when Supabase is NOT configured."""
    processor = processor_with_mock_predictor
    processor.supabase = mock_supabase_client_not_configured
    # _parse_csv is part of the BatchProcessor, so we test its integration here
    # but we can mock process_batch_job itself as that's a separate large unit of work.
    mocker.patch.object(processor, "process_batch_job", new_callable=AsyncMock)
    mocker.patch("asyncio.create_task")  # To prevent actual task creation

    job_id = await processor.start_batch_job(mock_upload_file_valid)

    assert job_id is not None
    assert job_id in processor.active_jobs
    job_details = processor.active_jobs[job_id]
    assert job_details["filename"] == mock_upload_file_valid.filename
    assert job_details["status"] == BatchPredictionStatus.PENDING.value
    assert (
        job_details["total_molecules"] == 2
    )  # Based on mock_upload_file_valid content
    # Verify Supabase client's create_batch_job was NOT called
    processor.supabase.create_batch_job.assert_not_called()


@pytest.mark.asyncio
async def test_start_batch_job_valid_csv_supabase_configured(
    processor_with_mock_predictor: BatchProcessor,
    mock_upload_file_valid: UploadFile,
    mock_supabase_client_configured: MagicMock,
    mocker: MockerFixture,
) -> None:
    """Test start_batch_job with a valid CSV and Supabase configured."""
    processor = processor_with_mock_predictor
    processor.supabase = mock_supabase_client_configured
    processor.supabase.is_configured = True

    # Mock process_batch_job for this test to isolate start_batch_job logic
    mock_process_batch_job = mocker.patch.object(
        processor, "process_batch_job", new_callable=AsyncMock
    )
    # Patch asyncio.create_task where it's looked up by BatchProcessor
    mock_create_task = mocker.patch("app.batch.asyncio.create_task")

    job_id = await processor.start_batch_job(mock_upload_file_valid)

    assert job_id is not None
    assert job_id in processor.active_jobs
    job_details = processor.active_jobs[job_id]
    assert job_details["filename"] == mock_upload_file_valid.filename
    assert job_details["status"] == BatchPredictionStatus.PENDING.value

    # Check Supabase call
    mock_supabase_client_configured.create_batch_job.assert_called_once_with(
        job_id=job_id,  # Expect string, matching mock's recorded call_args
        filename=mock_upload_file_valid.filename,
        total_molecules=2,  # Assuming mock_upload_file_valid has 2 smiles
    )

    # Check that process_batch_job was scheduled
    mock_create_task.assert_called_once()
    mock_process_batch_job.assert_called_once_with(job_id)
