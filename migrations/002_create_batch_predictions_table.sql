-- Migration: Create batch_predictions table for storing CSV batch prediction jobs
-- This table supports the /batch_predict_csv endpoint for processing multiple predictions

CREATE TABLE IF NOT EXISTS batch_predictions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    filename TEXT,
    total_molecules INTEGER DEFAULT 0,
    processed_molecules INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result_url TEXT,  -- URL to download the results CSV
    
    -- Add metadata for tracking
    user_agent TEXT,
    ip_address TEXT
);

-- Create the batch_prediction_items table to store individual results in a batch
CREATE TABLE IF NOT EXISTS batch_prediction_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_id UUID NOT NULL REFERENCES batch_predictions(id) ON DELETE CASCADE,
    smiles TEXT NOT NULL,
    probability REAL,
    model_version TEXT,
    error_message TEXT,
    row_number INTEGER NOT NULL,  -- Preserves original order in CSV
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
    
    -- For quick lookup of all items in a batch
    CONSTRAINT valid_probability CHECK (probability IS NULL OR (probability >= 0 AND probability <= 1))
);

-- Index on batch_id for quick lookups of items in a batch
CREATE INDEX IF NOT EXISTS idx_batch_prediction_items_batch_id ON batch_prediction_items (batch_id);

-- Add comments for documentation
COMMENT ON TABLE batch_predictions IS 'Stores batch prediction jobs from the /batch_predict_csv endpoint';
COMMENT ON TABLE batch_prediction_items IS 'Stores individual molecules from batch prediction jobs';
