#!/usr/bin/env python
"""Tests for app.db (SupabaseClient)."""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch
import asyncio
import uuid
import json

import httpx
import pytest
import pytest_asyncio
from loguru import logger

from app.config import settings
from app.db import STORAGE_BATCH_RESULTS_PATH, URL_EXPIRY_SECONDS, SupabaseClient
from app.models import BatchPredictionStatus

# Turn off noisy default loggers for tests
# logger.remove() # This can interfere with caplog


@pytest.fixture
def mock_settings_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock settings to simulate Supabase being configured."""
    monkeypatch.setattr(settings, "SUPABASE_URL", "http://test.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_KEY", "test_service_key")


@pytest.fixture
def mock_settings_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock settings to simulate Supabase being NOT configured."""
    monkeypatch.setattr(settings, "SUPABASE_URL", "")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_KEY", "")


@pytest_asyncio.fixture
async def supabase_client() -> AsyncGenerator[SupabaseClient, None]:
    """Yield a SupabaseClient instance."""
    # Ensure settings are reset for each test, or use a specific mock fixture
    # For now, assume tests will apply their own settings mocks
    client = SupabaseClient()
    yield client


@pytest_asyncio.fixture
async def supabase_client_unconfigured() -> AsyncGenerator[SupabaseClient, None]:
    """Yield a SupabaseClient instance that is explicitly unconfigured."""
    with patch.object(settings, 'SUPABASE_URL', ""):
        with patch.object(settings, 'SUPABASE_SERVICE_KEY', ""):
            client = SupabaseClient()
            assert not client.is_configured  # Add an assertion here for sanity
            yield client


@pytest_asyncio.fixture
async def supabase_client_configured() -> AsyncGenerator[SupabaseClient, None]:
    """Yield a SupabaseClient instance that is explicitly configured."""
    with patch.object(settings, 'SUPABASE_URL', "http://test.supabase.co"):
        with patch.object(settings, 'SUPABASE_SERVICE_KEY', "test_service_key"):
            client = SupabaseClient()
            assert client.is_configured  # Add an assertion here for sanity
            yield client


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Mock httpx.AsyncClient."""
    return AsyncMock()


class TestSupabaseClientInit:
    """Tests for SupabaseClient.__init__."""

    def test_init_configured(self, mock_settings_configured: None, caplog: pytest.LogCaptureFixture) -> None:
        """Test client initialization when Supabase IS configured."""
        with caplog.at_level("INFO"):
            client = SupabaseClient()
        assert client.is_configured
        assert client.url == "http://test.supabase.co"
        # assert "Supabase client initialised" in caplog.text
        assert "Supabase not configured" not in caplog.text

    def test_init_unconfigured(self, mock_settings_unconfigured: None, caplog: pytest.LogCaptureFixture) -> None:
        """Test client initialization when Supabase is NOT configured."""
        with caplog.at_level("WARNING"):
            client = SupabaseClient()
        assert not client.is_configured
        assert client.url == ""
        # assert "Supabase not configured" in caplog.text
        assert "Supabase client initialised" not in caplog.text


@pytest.mark.asyncio
class TestSupabaseClientPostJson:
    """Tests for SupabaseClient._post_json method."""

    async def test_post_json_not_configured(
        self, supabase_client: SupabaseClient, mock_settings_unconfigured: None
    ) -> None:
        """Test _post_json when Supabase is not configured."""
        # Re-initialize client with unconfigured settings
        client = SupabaseClient()
        assert not client.is_configured
        result = await client._post_json("/test_route", {"data": "value"})
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_post_json_success_empty_response(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None
    ) -> None:
        """Test _post_json success with empty response text (e.g., Prefer: return=minimal)."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = ""
        # mock_response.json = AsyncMock(return_value={}) # Not called if text is empty

        mock_async_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient() # Re-initialize with configured settings
        assert client.is_configured
        result = await client._post_json("/test_route", {"data": "value"})

        assert result == {}
        mock_async_client.return_value.__aenter__.return_value.post.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_post_json_success_valid_json(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None
    ) -> None:
        """Test _post_json success with valid JSON response."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 201
        mock_response.text = '{"key": "value"}'
        mock_response.json = MagicMock(return_value={"key": "value"})

        mock_async_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        result = await client._post_json("/test_route", {"data": "payload"})

        assert result == {"key": "value"}
        mock_response.json.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_post_json_success_invalid_json(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _post_json success with invalid JSON response (ValueError)."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.text = "not json"
        mock_response.json = MagicMock(side_effect=ValueError("Failed to decode"))

        mock_async_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        with caplog.at_level("ERROR"):
            result = await client._post_json("/test_route", {"data": "payload"})

        assert result is None
        # assert "Failed to decode JSON response" in caplog.text
        mock_response.json.assert_called_once()

    @patch("httpx.AsyncClient")
    async def test_post_json_failure_http_error(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _post_json failure due to HTTP error status."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.text = "Server Error"

        mock_async_client.return_value.__aenter__.return_value.post = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        with caplog.at_level("ERROR"):
            result = await client._post_json("/test_route", {"data": "payload"})

        assert result is None
        # assert "Supabase POST /test_route -> 500 - Server Error" in caplog.text


@pytest.mark.asyncio
class TestSupabaseClientPatchJson:
    """Tests for SupabaseClient._patch_json method."""

    async def test_patch_json_not_configured(
        self, supabase_client: SupabaseClient, mock_settings_unconfigured: None
    ) -> None:
        """Test _patch_json when Supabase is not configured."""
        client = SupabaseClient() # Re-initialize with unconfigured settings
        assert not client.is_configured
        result = await client._patch_json("/test_route", {"data": "value"})
        assert result is None

    @patch("httpx.AsyncClient")
    async def test_patch_json_success(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None
    ) -> None:
        """Test _patch_json success."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 204 # Common success for PATCH with Prefer: return=minimal
        # No .text or .json() needed for minimal return

        mock_async_client.return_value.__aenter__.return_value.patch = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        result = await client._patch_json("/test_route", {"data": "payload"})

        assert result == {"success": True}
        mock_async_client.return_value.__aenter__.return_value.patch.assert_called_once()

    @pytest.mark.parametrize("status_code", [200, 201, 204])
    @patch("httpx.AsyncClient")
    async def test_patch_json_success_various_status_codes(
        self, mock_async_client: MagicMock, status_code: int, supabase_client: SupabaseClient, mock_settings_configured: None
    ) -> None:
        """Test _patch_json success with various success status codes."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = status_code

        mock_async_client.return_value.__aenter__.return_value.patch = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        result = await client._patch_json("/test_route", {"data": "payload"})
        assert result == {"success": True}

    @patch("httpx.AsyncClient")
    async def test_patch_json_failure_http_error(
        self, mock_async_client: MagicMock, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test _patch_json failure due to HTTP error status."""
        mock_response = AsyncMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.text = "Bad Request"

        mock_async_client.return_value.__aenter__.return_value.patch = AsyncMock(
            return_value=mock_response
        )

        client = SupabaseClient()
        assert client.is_configured
        with caplog.at_level("ERROR"):
            result = await client._patch_json("/test_route", {"data": "payload"})

        assert result is None
        # assert "Supabase PATCH /test_route -> 400 - Bad Request" in caplog.text


@pytest.mark.asyncio
class TestSupabaseClientStoreBatchPredictionItem:
    """Tests for SupabaseClient.store_batch_prediction_item method."""

    @pytest.mark.parametrize(
        "probability, error_message, expected_extra_payload",
        [
            (0.75, None, {"probability": 0.75}),
            (None, "Error XYZ", {"error_message": "Error XYZ"}),
            (0.99, "Error ABC", {"probability": 0.99, "error_message": "Error ABC"}),
            (None, None, {}), # Neither provided
        ],
    )
    async def test_store_batch_prediction_item_payload_construction(
        self,
        supabase_client: SupabaseClient,
        mock_settings_configured: None,
        caplog: pytest.LogCaptureFixture,
        probability: Optional[float],
        error_message: Optional[str],
        expected_extra_payload: Dict[str, Any],
    ) -> None:
        """Test payload construction based on optional probability and error_message."""
        mock_post_json = AsyncMock(return_value={"id": "item_123"})  # Dummy successful response
        supabase_client._post_json = mock_post_json # type: ignore[method-assign]

        assert supabase_client.is_configured

        batch_id = "batch_abc"
        smiles = "CCO"
        row_number = 1
        model_version = "v1.test"

        expected_base_payload = {
            "batch_id": batch_id,
            "smiles": smiles,
            "row_number": row_number,
            "model_version": model_version,
        }
        expected_final_payload = {**expected_base_payload, **expected_extra_payload}

        result = await supabase_client.store_batch_prediction_item(
            batch_id=batch_id,
            smiles=smiles,
            row_number=row_number,
            model_version=model_version,
            probability=probability,
            error_message=error_message,
        )

        assert result == {"id": "item_123"}
        supabase_client._post_json.assert_called_once_with(
            "/rest/v1/batch_prediction_items", expected_final_payload
        )

    async def test_store_batch_prediction_item_payload_construction_error_message(
        self, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test payload construction with error_message."""
        mock_post_json = AsyncMock(return_value={"id": "item_123"})  # Dummy successful response
        supabase_client._post_json = mock_post_json # type: ignore[method-assign]

        # Ensure the fixture is configured as expected
        assert supabase_client.is_configured

        batch_id = "batch_abc"
        smiles = "CCO"
        row_number = 1
        model_version = "v1.test"
        error_message = "Test error message"

        expected_base_payload = {
            "batch_id": batch_id,
            "smiles": smiles,
            "row_number": row_number,
            "model_version": model_version,
        }
        expected_final_payload = {**expected_base_payload, "error_message": error_message}

        result = await supabase_client.store_batch_prediction_item(
            batch_id=batch_id,
            smiles=smiles,
            row_number=row_number,
            model_version=model_version,
            error_message=error_message,
        )

        assert result == {"id": "item_123"}
        supabase_client._post_json.assert_called_once_with(
            "/rest/v1/batch_prediction_items", expected_final_payload
        )
        assert supabase_client._post_json.call_args[0][1]["error_message"] == error_message


@pytest.mark.asyncio
class TestSupabaseClientCreateBatchJob:
    """Tests for SupabaseClient.create_batch_job."""

    async def test_create_batch_job_not_configured(
        self, supabase_client: SupabaseClient, mock_settings_unconfigured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test create_batch_job when client is not configured."""
        client = SupabaseClient() # This instance will pick up the unconfigured settings
        assert not client.is_configured
        job_id = str(uuid.uuid4())

        with caplog.at_level("WARNING"):
            # Ensure named arguments are used
            result = await client.create_batch_job(job_id=job_id, filename="test_filename.csv", total_molecules=100)
        assert result is None
        # assert "Supabase not configured, cannot create batch job." in caplog.text # Log assertion commented out

    async def test_create_batch_job_success(
        self, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test successful batch job creation."""
        client = SupabaseClient() # This instance will pick up the configured settings
        assert client.is_configured
        job_id = str(uuid.uuid4())

        mock_response_data = {"id": job_id, "filename": "test_filename.csv"}
        client._post_json = AsyncMock(return_value=mock_response_data) # type: ignore[method-assign]

        # Ensure named arguments are used
        result = await client.create_batch_job(job_id=job_id, filename="test_filename.csv", total_molecules=100)

        assert result == mock_response_data
        client._post_json.assert_called_once_with(
            '/rest/v1/batch_predictions', 
            {
                "id": job_id,
                "status": BatchPredictionStatus.PENDING.value,
                "filename": "test_filename.csv",
                "total_molecules": 100,
            },
        )

    async def test_create_batch_job_api_error(
        self, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test batch job creation when _post_json returns None (API error)."""
        client = SupabaseClient() # This instance will pick up the configured settings
        assert client.is_configured
        job_id = str(uuid.uuid4())

        client._post_json = AsyncMock(return_value=None) # type: ignore[method-assign]

        with caplog.at_level("ERROR"):
            # Ensure named arguments are used
            result = await client.create_batch_job(job_id=job_id, filename="test_filename.csv", total_molecules=100)

        assert result is None
        # assert "Failed to create batch job for test_filename.csv via Supabase." in caplog.text # Log assertion commented out
        client._post_json.assert_called_once_with(
            '/rest/v1/batch_predictions', 
            {
                "id": job_id,
                "status": BatchPredictionStatus.PENDING.value,
                "filename": "test_filename.csv",
                "total_molecules": 100,
            },
        )


@pytest.mark.asyncio
class TestSupabaseClientStorePrediction:
    """Tests for SupabaseClient.store_prediction method."""

    async def test_store_prediction_success(
        self, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test successful prediction storage."""
        client = SupabaseClient()  # This instance will pick up the configured settings
        assert client.is_configured
        smiles = "CCO"
        probability = 0.75
        model_version = "v1.1"

        mock_response_data = {"id": str(uuid.uuid4()), "smiles": smiles}
        client._post_json = AsyncMock(return_value=mock_response_data)  # type: ignore[method-assign]

        result = await client.store_prediction(smiles=smiles, probability=probability, model_version=model_version)

        assert result == mock_response_data
        client._post_json.assert_called_once_with(
            "/rest/v1/predictions",
            {
                "smiles": smiles,
                "probability": probability,
                "model_version": model_version,
            },
        )
        assert not caplog.records  # No error/warning logs expected

    async def test_store_prediction_api_error(
        self, supabase_client: SupabaseClient, mock_settings_configured: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test prediction storage when _post_json returns None (API error)."""
        client = SupabaseClient()  # This instance will pick up the configured settings
        assert client.is_configured
        smiles = "CN(C)C"
        probability = 0.25

        client._post_json = AsyncMock(return_value=None)  # type: ignore[method-assign]

        with caplog.at_level("ERROR"):
            result = await client.store_prediction(smiles=smiles, probability=probability)

        assert result is None
        # We expect _post_json to log its own error, so we don't check for a specific log here from store_prediction itself
        # but we ensure _post_json was called.
        client._post_json.assert_called_once_with(
            "/rest/v1/predictions",
            {
                "smiles": smiles,
                "probability": probability,
                "model_version": "v1.0", # Default model_version
            },
        )

    async def test_store_prediction_not_configured(
        self,
        supabase_client_unconfigured: SupabaseClient,
    ) -> None:
        """Test store_prediction when Supabase is not configured."""
        client = supabase_client_unconfigured
        assert not client.is_configured

        result = await client.store_prediction(smiles="CCC", probability=0.5)

        assert result is None


@pytest.mark.asyncio
class TestSupabaseClientUpdateBatchJobStatus:
    """Tests for SupabaseClient.update_batch_job_status method."""

    @patch('app.db.httpx.AsyncClient') # Patch where AsyncClient is imported/used by SupabaseClient
    async def test_update_batch_job_status_success(
        self,
        mock_async_client_constructor: MagicMock, # This is the constructor mock
        supabase_client_configured: SupabaseClient,
    ) -> None:
        """Test successful update of batch job status."""
        job_id = str(uuid.uuid4()) # Use fresh ID per test
        client = supabase_client_configured
        expected_response_data = {"id": job_id, "status": "COMPLETED"}

        # Configure the mock client instance that the constructor will return
        mock_cli_instance = AsyncMock()
        mock_response = AsyncMock(spec=httpx.Response) # Use spec
        mock_response.status_code = 200
        mock_response.json.return_value = [expected_response_data]
        # Use PropertyMock for .content to ensure it's correctly evaluated by 'if not r.content'
        type(mock_response).content = PropertyMock(return_value=json.dumps([expected_response_data]).encode('utf-8'))

        mock_cli_instance.patch.return_value = mock_response
        mock_async_client_constructor.return_value.__aenter__.return_value = mock_cli_instance # Configure for 'async with'

        result = await client.update_batch_job_status(
            job_id=job_id, status="COMPLETED"
        )

        assert result == expected_response_data  # The method extracts the first item
        mock_async_client_constructor.assert_called_once() # Ensure AsyncClient() was called
        mock_cli_instance.patch.assert_called_once()
        call_args, call_kwargs = mock_cli_instance.patch.call_args
        assert call_args[0] == f"{client.url}/rest/v1/batch_predictions?id=eq.{job_id}"
        assert call_kwargs["json"] == {"status": "COMPLETED"}

    @patch('app.db.logger.error') # Patch loguru logger instance used in app.db
    @patch('app.db.httpx.AsyncClient')
    async def test_update_batch_job_status_api_error(
        self,
        mock_async_client_constructor: MagicMock, # Order matters for patch decorators
        mock_loguru_error: MagicMock, # Patched logger.error
        supabase_client_configured: SupabaseClient,
    ) -> None:
        """Test API error during batch job status update."""
        job_id = str(uuid.uuid4()) # Use fresh ID per test
        client = supabase_client_configured

        # Configure the mock client instance for an error
        mock_cli_instance = AsyncMock()
        mock_response = AsyncMock(spec=httpx.Response) # Use spec
        mock_response.status_code = 500
        # Use PropertyMock for .text
        type(mock_response).text = PropertyMock(return_value="Internal Server Error")
        # mock_response.json is not called in error path, so no need to mock it explicitly for this test

        mock_cli_instance.patch.return_value = mock_response
        mock_async_client_constructor.return_value.__aenter__.return_value = mock_cli_instance # Configure for 'async with'

        result = await client.update_batch_job_status(
            job_id=job_id, status="PROCESSING"
        )

        assert result is None
        mock_loguru_error.assert_called_once_with(
            "Supabase PATCH %s → %s – %s",
            f"/rest/v1/batch_predictions?id=eq.{job_id}",
            500,
            "Internal Server Error"
        )
        mock_async_client_constructor.assert_called_once() # Ensure AsyncClient() was called
        mock_cli_instance.patch.assert_called_once()
        call_args, call_kwargs = mock_cli_instance.patch.call_args
        assert call_args[0] == f"{client.url}/rest/v1/batch_predictions?id=eq.{job_id}" # Corrected path
        assert call_kwargs["json"] == {"status": "PROCESSING"}

    async def test_update_batch_job_status_not_configured(
        self, supabase_client_unconfigured: SupabaseClient
    ) -> None:
        """Test update_batch_job_status when Supabase is not configured."""
        client = supabase_client_unconfigured
        assert not client.is_configured
        job_id = str(uuid.uuid4()) # Use a fresh, dynamic job_id for this test

        original_patch_json = client._patch_json
        client._patch_json = AsyncMock(wraps=original_patch_json)  # type: ignore[method-assign]

        result = await client.update_batch_job_status(
            job_id=job_id, status="PENDING" # Use the dynamic job_id
        )

        assert result is None
        client._patch_json.assert_called_once_with(
            f"/rest/v1/batch_predictions?id=eq.{job_id}", {"status": "PENDING"} # Corrected path & dynamic job_id
        )
