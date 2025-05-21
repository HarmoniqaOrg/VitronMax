import pytest
import io
from fastapi import UploadFile
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from pytest_mock import MockerFixture
import pytest_asyncio

from app.main import app
from app.models import BatchPredictionStatus
from app.batch import BatchProcessor
from app.predict import BBBPredictor
from app.db import SupabaseClient


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def mock_upload_file_valid() -> MagicMock:
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test_smiles.csv"
    smiles_content = "SMILES\nCCO\nCCC"
    mock_file.file = io.StringIO(smiles_content)
    mock_file.read = AsyncMock(return_value=smiles_content.encode("utf-8"))
    mock_file.file.seek(0)
    return mock_file


@pytest.fixture
def mock_upload_file_invalid_format() -> MagicMock:
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "invalid_format.txt"
    invalid_content = "This is not a CSV file."
    mock_file.file = io.StringIO(invalid_content)
    mock_file.read = AsyncMock(return_value=invalid_content.encode("utf-8"))
    mock_file.file.seek(0)
    return mock_file


@pytest_asyncio.fixture
async def mock_supabase_client_configured(mocker: MockerFixture) -> AsyncMock:
    mock_client: AsyncMock = mocker.AsyncMock(spec=SupabaseClient)
    mock_client.is_configured = True
    mock_client.create_batch_job = AsyncMock(return_value={"id": "test_job_id"})
    mock_client.get_batch_job_status = AsyncMock(
        return_value={"status": BatchPredictionStatus.COMPLETED.value}
    )
    mock_client.get_batch_job_results_csv_url = AsyncMock(
        return_value="mock_signed_url_completed.csv"
    )
    mock_client.store_batch_result_csv = AsyncMock(
        return_value="mock_storage_url/some_job.csv"
    )
    mock_client.update_batch_job_status = AsyncMock()
    mock_client.update_batch_job_results = AsyncMock()
    return mock_client


@pytest_asyncio.fixture
async def mock_supabase_client_not_configured(mocker: MockerFixture) -> AsyncMock:
    mock_client: AsyncMock = mocker.AsyncMock(spec=SupabaseClient)
    mock_client.is_configured = False
    mock_client.create_batch_job = AsyncMock(return_value=None)
    mock_client.get_batch_job_status = AsyncMock(return_value=None)
    mock_client.get_batch_job_results_csv_url = AsyncMock(return_value=None)
    mock_client.store_batch_result_csv = AsyncMock(return_value=None)
    mock_client.update_batch_job_status = AsyncMock()
    mock_client.update_batch_job_results = AsyncMock()
    return mock_client


@pytest_asyncio.fixture
async def processor_with_mock_predictor(mocker: MockerFixture) -> BatchProcessor:
    """Fixture to provide a BatchProcessor with a mocked predictor and Supabase client."""
    mock_predictor = mocker.AsyncMock(spec=BBBPredictor)
    mock_predictor.version = "mock_v1.0"
    mock_supabase_client = mocker.AsyncMock(spec=SupabaseClient)
    mock_supabase_client.is_configured = True
    mock_supabase_client.create_batch_job = AsyncMock(
        return_value={"id": "test_job_id"}
    )
    mock_supabase_client.store_batch_result_csv = AsyncMock(
        return_value="mock_storage_url/default_job.csv"
    )
    mock_supabase_client.update_batch_job_status = AsyncMock()
    mock_supabase_client.update_batch_job_results = AsyncMock()
    return BatchProcessor(
        predictor=mock_predictor, supabase_client=mock_supabase_client
    )


@pytest.fixture(autouse=True)
def mock_global_batch_processor(
    mocker: MockerFixture, processor_with_mock_predictor: BatchProcessor
) -> None:
    """Automatically mock the global batch_processor instance in app.main."""
    mocker.patch("app.main.batch_processor", new=processor_with_mock_predictor)
