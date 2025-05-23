# scripts/supabase_sanity_probe.py
import os
import sys
import requests

# Adjust path to import config from the app directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)

try:
    from app.config import settings
except ImportError as e:
    print(f"Error importing app.config: {e}")
    print(
        "Ensure this script is run from the project root or the PYTHONPATH is set correctly."
    )
    sys.exit(1)

PROBE_RESULTS = {
    "predictions_count": {"status": "", "details": ""},
    "batch_predictions_limit1": {"status": "", "details": ""},
    "storage_bucket_check": {"status": "", "details": ""},
}


def run_supabase_probes() -> None:
    """Runs read-only sanity checks against Supabase."""
    print("Starting Supabase sanity probes...")

    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        print("Error: SUPABASE_URL or SUPABASE_SERVICE_KEY is not set.")
        print("Please ensure they are configured in your .env file or environment.")
        PROBE_RESULTS["predictions_count"]["status"] = "Error"
        PROBE_RESULTS["predictions_count"]["details"] = "Supabase URL/Key not set"
        PROBE_RESULTS["batch_predictions_limit1"]["status"] = "Error"
        PROBE_RESULTS["batch_predictions_limit1"][
            "details"
        ] = "Supabase URL/Key not set"
        PROBE_RESULTS["storage_bucket_check"]["status"] = "Error"
        PROBE_RESULTS["storage_bucket_check"]["details"] = "Supabase URL/Key not set"
        return

    headers = {
        "apikey": settings.SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
    }

    # 1. Check predictions count
    try:
        url_preds_count = f"{settings.SUPABASE_URL}/rest/v1/predictions?select=count"
        response = requests.get(url_preds_count, headers=headers, timeout=10)
        if response.status_code == 200:
            count = response.json()[0]["count"]
            PROBE_RESULTS["predictions_count"]["status"] = "OK"
            PROBE_RESULTS["predictions_count"]["details"] = f"Count: {count}"
        else:
            PROBE_RESULTS["predictions_count"][
                "status"
            ] = f"Error ({response.status_code})"
            PROBE_RESULTS["predictions_count"]["details"] = response.text[:100]
    except requests.RequestException as e:
        PROBE_RESULTS["predictions_count"]["status"] = "Request Failed"
        PROBE_RESULTS["predictions_count"]["details"] = str(e)[:100]

    # 2. Check batch_predictions limit 1
    try:
        url_batch_preds = (
            f"{settings.SUPABASE_URL}/rest/v1/batch_predictions?select=id&limit=1"
        )
        response = requests.get(url_batch_preds, headers=headers, timeout=10)
        if response.status_code == 200:
            PROBE_RESULTS["batch_predictions_limit1"]["status"] = "OK"
            PROBE_RESULTS["batch_predictions_limit1"][
                "details"
            ] = f"{len(response.json())} record(s) returned (expected 0 or 1)"
        else:
            PROBE_RESULTS["batch_predictions_limit1"][
                "status"
            ] = f"Error ({response.status_code})"
            PROBE_RESULTS["batch_predictions_limit1"]["details"] = response.text[:100]
    except requests.RequestException as e:
        PROBE_RESULTS["batch_predictions_limit1"]["status"] = "Request Failed"
        PROBE_RESULTS["batch_predictions_limit1"]["details"] = str(e)[:100]

    # 3. Check storage bucket existence
    try:
        url_buckets = f"{settings.SUPABASE_URL}/storage/v1/bucket"
        response = requests.get(url_buckets, headers=headers, timeout=10)
        if response.status_code == 200:
            buckets = response.json()
            bucket_found = any(
                b["name"] == settings.STORAGE_BUCKET_NAME for b in buckets
            )
            if bucket_found:
                PROBE_RESULTS["storage_bucket_check"]["status"] = "OK"
                PROBE_RESULTS["storage_bucket_check"][
                    "details"
                ] = f"Bucket '{settings.STORAGE_BUCKET_NAME}' found."
            else:
                PROBE_RESULTS["storage_bucket_check"]["status"] = "Not Found"
                PROBE_RESULTS["storage_bucket_check"][
                    "details"
                ] = f"Bucket '{settings.STORAGE_BUCKET_NAME}' not found."
        else:
            PROBE_RESULTS["storage_bucket_check"][
                "status"
            ] = f"Error ({response.status_code})"
            PROBE_RESULTS["storage_bucket_check"]["details"] = response.text[:100]
    except requests.RequestException as e:
        PROBE_RESULTS["storage_bucket_check"]["status"] = "Request Failed"
        PROBE_RESULTS["storage_bucket_check"]["details"] = str(e)[:100]

    print("Supabase sanity probes completed.")


if __name__ == "__main__":
    run_supabase_probes()
    print("\n--- Supabase Sanity Probe Results ---")
    for probe, result in PROBE_RESULTS.items():
        print(
            f"{probe.replace('_', ' ').title()}: [{result['status']}] - {result['details']}"
        )
    print("-------------------------------------")
