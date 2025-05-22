# VitronMax

In-silico Blood-Brain Barrier (BBB) permeability prediction API with Random Forest model, batch processing, and PDF reporting.

## Overview

VitronMax is a FastAPI-based service that predicts blood-brain barrier permeability for drug candidates using a trained random forest model based on Morgan fingerprints (2048 bits, radius 2). The API provides:

- SMILES-based BBB permeability prediction with Morgan fingerprints via RDKit
- Batch processing of CSV files with asynchronous job tracking
- PDF report generation with molecule visualization and interpretation
- Data persistence through Supabase (PostgreSQL + Storage)
- Containerized deployment with Docker
- CI/CD through GitHub Actions with quality gates (ruff, black, mypy --strict)
- Health check endpoint for monitoring and deployment validation

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- Git
- RDKit dependencies (for local non-Docker development)
  - On Ubuntu/Debian: `apt-get install libglib2.0-0 libxrender1 libsm6 libxext6`
  - On macOS: `brew install cairo pango glib`

### Local Development

1. Clone the repository
   ```bash
   git clone https://github.com/YourOrg/VitronMax.git
   cd VitronMax
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables
   ```bash
   cp .env.example .env
   # Edit .env with your specific settings
   ```

5. Run the development server
   ```bash
   uvicorn app.main:app --reload
   ```

6. Access the API documentation at http://localhost:8000/docs

### Docker

Build and run the application in a Docker container:

```bash
docker build -t vitronmax:latest .
docker run -p 8080:8080 vitronmax:latest
```

Access the API at http://localhost:8080/docs

## API Documentation

Detailed API documentation is available in the [api-documentation.md](./docs/api-documentation.md) file.

### Key Endpoints

- `GET /`: Service information
- `GET /healthz`: Health check endpoint for monitoring
- `POST /predict_fp`: Predict BBB permeability from SMILES string
- `POST /batch_predict_csv`: Process a batch of SMILES from a CSV file
- `GET /batch_status/{job_id}`: Check the status of a batch job
- `GET /download/{job_id}`: Download batch prediction results
- `POST /report`: Generate a PDF report for a single molecule

### Example Usage

#### Single Molecule Prediction
```bash
curl -X POST "http://localhost:8080/predict_fp" \
  -H "Content-Type: application/json" \
  -d '{"smi": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

#### Batch Processing
```bash
# Submit a batch prediction job
curl -X POST "http://localhost:8080/batch_predict_csv" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@molecules.csv" \
  

# Check job status
curl -X GET "http://localhost:8080/batch_status/your-job-id"

# Download results when job is completed
curl -X GET "http://localhost:8080/download/your-job-id" -o results.csv
```

#### Generate PDF Report
```bash
curl -X POST "http://localhost:8080/report" \
  -H "Content-Type: application/json" \
  -d '{"smi": "CC(=O)OC1=CC=CC=C1C(=O)O"}' \
  -o molecule_report.pdf
```

## Supabase Configuration

### Storage Setup

VitronMax uses Supabase Storage to reliably persist batch prediction results. The application will automatically create the required storage bucket (default name: `vitronmax`) on startup if it doesn't exist.

To verify the Supabase Storage functionality:

1. Ensure your `.env` file contains valid Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url_here
   SUPABASE_SERVICE_KEY=your_supabase_service_key_here
   STORAGE_BUCKET_NAME=vitronmax  # Optional, defaults to 'vitronmax'
   ```

2. Submit a batch prediction job with a CSV file containing SMILES strings

3. Once the job completes, check the status endpoint - it should include a signed URL to download the results from Supabase Storage

4. The download endpoint will automatically redirect to the Supabase Storage signed URL if available

Alternatively, you can use the `scripts/supabase_sanity_probe.py` script to directly check Supabase connectivity and bucket status:
```bash
# Ensure .env is configured with SUPABASE_URL and SUPABASE_SERVICE_KEY
python scripts/supabase_sanity_probe.py
```

### Database Tables

VitronMax requires the following Supabase tables. The schema is defined in `db/schema.sql`:

- `predictions`: Stores individual prediction results
- `batch_predictions`: Stores batch job metadata
- `batch_prediction_items`: Stores individual results within a batch job

## Deployment

### Fly.io Deployment

1. Install the Fly CLI: https://fly.io/docs/hands-on/install-flyctl/

2. Login to Fly
   ```bash
   flyctl auth login
   ```

3. Set up your Fly.io app (first time only)
   ```bash
   flyctl launch
   ```

4. Set environment secrets
   ```bash
   flyctl secrets set SUPABASE_URL=your_url SUPABASE_SERVICE_KEY=your_key
   flyctl secrets set OPENAI_API_KEY=your_openai_key
   ```

5. Deploy the application
   ```bash
   flyctl deploy
   ```

## Development Workflow

1. Create a feature branch from `main` for each feature
2. Implement changes and confirm they pass all tests
3. Ensure documentation is updated and code is formatted with `black`
4. Open a pull request with reference to PRD bullet points
5. Wait for CI checks to pass and code review

## Testing

Run the full local quality suite (linting, type checking, tests):
```bash
ruff check .
black --check .
mypy app/ --strict
pytest
```

For coverage report:

```bash
pytest --cov=app tests/
```

Local Docker test:

```bash
docker build . -t vitronmax:test
docker run -p 8080:8080 vitronmax:test pytest
```

## Project Structure

```
VitronMax/
├── .github/workflows/  # CI configuration
├── app/                # Application code
│   ├── __init__.py
│   ├── config.py       # Environment configuration
│   ├── db.py           # Supabase integration
│   ├── main.py         # FastAPI application
│   ├── models.py       # Pydantic models
│   └── predict.py      # BBB prediction logic
├── docs/               # Documentation
├── models/             # Trained ML models
├── tests/              # Test suite
├── .env.example        # Example environment variables
├── Dockerfile          # Container definition
├── fly.toml            # Fly.io configuration
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## License

MIT License
