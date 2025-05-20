"""
Supabase integration for VitronMax – async REST helper.

Lint-clean (ruff 0.3.x), black-formatted and mypy-strict compatible.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

import httpx
from loguru import logger

from app.config import settings
from app.models import BatchPredictionStatus

# ───────────────────────────────────────────────────────────
STORAGE_BATCH_RESULTS_PATH = "batch_results"
URL_EXPIRY_SECONDS = 60 * 60 * 24 * 7  # 7 days
# ───────────────────────────────────────────────────────────


class SupabaseClient:
    """Minimal async wrapper around Supabase PostgREST + Storage APIs."""

    def __init__(self) -> None:
        self.url: str = settings.SUPABASE_URL or ""
        self._api_key: str = settings.SUPABASE_SERVICE_KEY or ""
        self.is_configured: bool = bool(self.url and self._api_key)

        if not self.is_configured:
            logger.warning("Supabase not configured – DB operations skipped.")
        else:
            logger.info("Supabase client initialised")

    # ───────────────────────── helpers ──────────────────────────
    def _hdr(
        self, *, json_ct: bool = False, prefer_minimal: bool = False
    ) -> Dict[str, str]:
        """Return headers with **no Optional values** (mypy-safe)."""
        hdr: Dict[str, str] = {
            "apikey": self._api_key,
            "Authorization": f"Bearer {self._api_key}",
        }
        if json_ct:
            hdr["Content-Type"] = "application/json"
        if prefer_minimal:
            hdr["Prefer"] = "return=minimal"
        return hdr

    async def _post_json(
        self, route: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.is_configured:
            return None
        async with httpx.AsyncClient() as cli:
            r = await cli.post(
                f"{self.url}{route}",
                headers=self._hdr(json_ct=True, prefer_minimal=True),
                json=payload,
                timeout=5.0,
            )
            if r.status_code in (200, 201):
                # PostgREST can return empty body with 201+Prefer=return=minimal
                return cast(Dict[str, Any], r.json() or {})
            logger.error("Supabase POST %s → %s – %s", route, r.status_code, r.text)
            return None

    async def _patch_json(
        self, route: str, payload: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not self.is_configured:
            return None
        async with httpx.AsyncClient() as cli:
            r = await cli.patch(
                f"{self.url}{route}",
                headers=self._hdr(json_ct=True, prefer_minimal=True),
                json=payload,
                timeout=5.0,
            )
            if r.status_code in (200, 201, 204):
                return {"success": True}
            logger.error("Supabase PATCH %s → %s – %s", route, r.status_code, r.text)
            return None

    # ───────────────────── predictions table ───────────────────
    async def store_prediction(
        self, smiles: str, probability: float, model_version: str = "v1.0"
    ) -> Optional[Dict[str, Any]]:
        return await self._post_json(
            "/rest/v1/predictions",
            {
                "smiles": smiles,
                "probability": probability,
                "model_version": model_version,
            },
        )

    # ───────────────────── batch jobs table ────────────────────
    async def create_batch_job(
        self, job_id: str, filename: str | None, total_molecules: int
    ) -> Optional[Dict[str, Any]]:
        return await self._post_json(
            "/rest/v1/batch_predictions",
            {
                "id": job_id,
                "status": BatchPredictionStatus.PENDING.value,
                "filename": filename,
                "total_molecules": total_molecules,
            },
        )

    async def update_batch_job_status(
        self, job_id: str, status: str
    ) -> Optional[Dict[str, Any]]:
        return await self._patch_json(
            f"/rest/v1/batch_predictions?id=eq.{job_id}", {"status": status}
        )

    async def update_batch_job_progress(
        self, job_id: str, processed_molecules: int
    ) -> Optional[Dict[str, Any]]:
        return await self._patch_json(
            f"/rest/v1/batch_predictions?id=eq.{job_id}",
            {"processed_molecules": processed_molecules},
        )

    async def store_batch_prediction_item(
        self,
        batch_id: str,
        smiles: str,
        row_number: int,
        model_version: str,
        probability: float | None = None,
        error_message: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "batch_id": batch_id,
            "smiles": smiles,
            "row_number": row_number,
            "model_version": model_version,
        }
        if probability is not None:
            payload["probability"] = probability
        if error_message:
            payload["error_message"] = error_message
        return await self._post_json("/rest/v1/batch_prediction_items", payload)

    async def complete_batch_job(
        self, job_id: str, result_url: str
    ) -> Optional[Dict[str, Any]]:
        return await self._patch_json(
            f"/rest/v1/batch_predictions?id=eq.{job_id}",
            {
                "status": BatchPredictionStatus.COMPLETED.value,
                "completed_at": "now()",
                "result_url": result_url,
            },
        )

    async def fail_batch_job(
        self, job_id: str, error_message: str
    ) -> Optional[Dict[str, Any]]:
        return await self._patch_json(
            f"/rest/v1/batch_predictions?id=eq.{job_id}",
            {
                "status": BatchPredictionStatus.FAILED.value,
                "error_message": error_message,
                "completed_at": "now()",
            },
        )

    # ──────────────────── storage (CSV results) ─────────────────
    async def _bucket_exists(self, bucket: str) -> bool:
        async with httpx.AsyncClient() as cli:
            r = await cli.get(
                f"{self.url}/storage/v1/bucket", headers=self._hdr(), timeout=5.0
            )
            return r.status_code == 200 and any(b["name"] == bucket for b in r.json())

    async def _create_bucket(self, bucket: str) -> bool:
        async with httpx.AsyncClient() as cli:
            r = await cli.post(
                f"{self.url}/storage/v1/bucket",
                headers=self._hdr(json_ct=True),
                json={"name": bucket, "public": False},
                timeout=5.0,
            )
            return r.status_code in (200, 201)

    async def ensure_storage_bucket(self) -> bool:
        """Ensure a private Storage bucket exists."""
        if not self.is_configured:
            return False
        bucket = settings.STORAGE_BUCKET_NAME
        return (await self._bucket_exists(bucket)) or (
            await self._create_bucket(bucket)
        )

    async def store_batch_result_csv(
        self, job_id: str, csv_content: str
    ) -> Optional[str]:
        """Upload CSV & return signed URL valid for 7 days."""
        if not self.is_configured or not await self.ensure_storage_bucket():
            return None

        bucket = settings.STORAGE_BUCKET_NAME
        path = f"{STORAGE_BATCH_RESULTS_PATH}/{job_id}.csv"

        async with httpx.AsyncClient() as cli:
            # Upload
            up = await cli.post(
                f"{self.url}/storage/v1/object/{bucket}/{path}",
                headers={
                    "apikey": self._api_key,
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "text/csv",
                },
                content=csv_content.encode(),
                timeout=10.0,
            )
            if up.status_code not in (200, 201):
                logger.error("Upload failed → %s %s", up.status_code, up.text)
                return None

            # Sign
            sign = await cli.post(
                f"{self.url}/storage/v1/object/sign/{bucket}/{path}",
                headers=self._hdr(json_ct=True),
                json={"expiresIn": URL_EXPIRY_SECONDS},
                timeout=5.0,
            )
            if sign.status_code in (200, 201):
                rel_url: str = cast(Dict[str, Any], sign.json()).get("signedURL", "")
                if rel_url and not rel_url.startswith("http"):
                    rel_url = f"{self.url}{rel_url}"
                return rel_url

            logger.error("Signed-URL failed → %s %s", sign.status_code, sign.text)
            return None


# Global singleton (import-time creation is fine for FastAPI)
supabase = SupabaseClient()
