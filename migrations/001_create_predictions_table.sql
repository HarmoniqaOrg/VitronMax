-- Migration: Create predictions table for storing individual predictions
-- This table stores individual molecule predictions from the /predict_fp endpoint

CREATE TABLE IF NOT EXISTS predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    smiles TEXT NOT NULL,
    probability REAL NOT NULL CHECK (probability >= 0 AND probability <= 1),
    model_version TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    ip_address TEXT,  -- Optional, for rate limiting if needed
    request_id TEXT,  -- For grouping related predictions (optional)
    
    -- Add indices for common queries
    CONSTRAINT valid_probability CHECK (probability >= 0 AND probability <= 1)
);

-- Index on SMILES for quick lookups of previous predictions
CREATE INDEX IF NOT EXISTS idx_predictions_smiles ON predictions (smiles);

-- Index on created_at for time-based queries
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);

-- Add comment for documentation
COMMENT ON TABLE predictions IS 'Stores blood-brain barrier permeability predictions';
