# API Documentation

## Authentication
No authentication is required for the MVP phase.

## Endpoints

### GET /
Returns basic API information.

**Response**:
```json
{
  "message": "Welcome to VitronMax API",
  "version": "1.0"
}
```

### POST /predict_fp
Predicts blood-brain barrier permeability probability using Random Forest model.

**Request**:
```json
{
  "smi": "CCO"  // SMILES string of the molecule
}
```

**Response**:
```json
{
  "prob": 0.78,  // Probability of BBB permeability
  "version": "1.0"  // Model version
}
```

**Error responses**:
- 400: Invalid SMILES string
- 422: Validation error (missing or empty SMILES)
```

### POST /batch_predict_csv
Process a batch of SMILES strings from a CSV file for BBB permeability prediction. This is an asynchronous operation that returns a job ID for tracking progress.

**Request**:
- Multipart form with a CSV file upload
- The CSV file must have a header row with a column named 'SMILES', 'smi', or 'smile'

**Response** (Status code: 202 Accepted):
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",  // Job UUID
  "status": "pending",  // Job status (pending, processing, completed, failed)
  "filename": "molecules.csv",  // Original filename
  "total_molecules": 42,  // Total molecules to process
  "processed_molecules": 0,  // Number processed so far
  "created_at": "2025-05-19T20:00:00",  // Timestamp
  "completed_at": null,  // Completion timestamp (when available)
  "result_url": null  // URL to download results (when available)
}
```

**Error responses**:
- 400: Invalid CSV format or no valid SMILES found
- 422: Validation error (missing file)

### GET /batch_status/{job_id}
Check the status of a batch prediction job.

**Parameters**:
- `job_id`: UUID of the batch prediction job

**Response**:
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",  // Job UUID
  "status": "processing",  // Current status
  "progress": 45.5,  // Percentage complete (0-100)
  "filename": "molecules.csv",  // Original filename
  "created_at": "2025-05-19T20:00:00",  // Timestamp
  "result_url": null,  // URL to download results if completed
  "error_message": null  // Error message if failed
}
```

**Error responses**:
- 404: Job not found

### GET /download/{job_id}
Download the results of a completed batch prediction job as a CSV file.

**Parameters**:
- `job_id`: UUID of the completed batch prediction job

**Response**:
- CSV file download with predictions (Content-Type: text/csv)

**Error responses**:
- 400: Job not completed yet
- 404: Job not found or results not available

### POST /report
Generate a PDF report for a molecule based on its SMILES string.

**Request**:
```json
{
  "smi": "CCO"  // SMILES string of the molecule
}
```

**Response**:
- PDF file download with detailed report (Content-Type: application/pdf)

**Error responses**:
- 400: Invalid SMILES string
- 422: Validation error (missing or empty SMILES)
- 500: Error generating PDF report
