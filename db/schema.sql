-- VitronMax Supabase Database Schema
-- This file helps ensure the database tables match our implementation

-- Enable Extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Predictions Table
-- Stores individual BBB permeability predictions
CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    smiles TEXT NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create index on SMILES for faster lookups
CREATE INDEX IF NOT EXISTS idx_predictions_smiles ON predictions(smiles);

-- Batch Predictions Table
-- Stores metadata for batch prediction jobs
CREATE TABLE IF NOT EXISTS batch_predictions (
    id UUID PRIMARY KEY,
    status TEXT NOT NULL,
    filename TEXT,
    total_molecules INTEGER,
    processed_molecules INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    completed_at TIMESTAMP WITH TIME ZONE,
    result_url TEXT,
    error_message TEXT
);

-- Batch Prediction Items Table
-- Stores individual prediction results within a batch
CREATE TABLE IF NOT EXISTS batch_prediction_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    batch_id UUID NOT NULL REFERENCES batch_predictions(id),
    smiles TEXT NOT NULL,
    row_number INTEGER NOT NULL,
    probability DOUBLE PRECISION,
    model_version TEXT NOT NULL,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    UNIQUE(batch_id, row_number)
);

-- Create index on batch_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id ON batch_prediction_items(batch_id);

-- RLS (Row Level Security) Policies
-- Uncomment and customize these when implementing user authentication

-- ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE batch_predictions ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE batch_prediction_items ENABLE ROW LEVEL SECURITY;

-- CREATE POLICY "Allow read access to predictions" ON predictions
--     FOR SELECT USING (true);

-- CREATE POLICY "Allow read access to batch_predictions" ON batch_predictions
--     FOR SELECT USING (true);

-- CREATE POLICY "Allow read access to batch_prediction_items" ON batch_prediction_items
--     FOR SELECT USING (true);

-- Storage initialization is handled via API calls in the application code
-- Bucket name: vitronmax
-- Files stored as: batch_results/{job_id}.csv
