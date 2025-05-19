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
