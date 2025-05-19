# VitronMax

In-silico Blood-Brain Barrier (BBB) permeability prediction API with Random Forest model, SwissADME panel, and PDF reporting.

## Overview

VitronMax is a FastAPI-based service that predicts blood-brain barrier permeability for drug candidates using a trained random forest model. The API provides:

- SMILES-based BBB permeability prediction
- Logging and persistence through Supabase integration
- Containerized deployment with Docker
- CI/CD through GitHub Actions

## Getting Started

### Prerequisites

- Python 3.10+
- Docker
- Git

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
- `POST /predict_fp`: Predict BBB permeability from SMILES string

### Example Usage

```bash
curl -X POST "http://localhost:8080/predict_fp" \
  -H "Content-Type: application/json" \
  -d '{"smi": "CC(=O)OC1=CC=CC=C1C(=O)O"}'
```

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

Run the test suite with:

```bash
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
