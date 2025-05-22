--------------------------------------------------------------------------------
-- schema config
--
-- unlike _00_instantiate-schema.sql, this script will not be set to delete
--   or reset any critical items
--------------------------------------------------------------------------------

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_namespace WHERE nspname = 'atp'
    ) THEN
        RAISE EXCEPTION 'Schema "atp" does not exist';
    END IF;
END $$;

SET search_path TO atp;

--------------------------------------------------------------------------------
-- table creation statement
--------------------------------------------------------------------------------

CREATE TABLE prediction (
    -- identifier columns
    imdb_id VARCHAR(10) PRIMARY KEY CHECK (imdb_id ~ '^tt[0-9]{7,8}$'),
    -- output value columns
    prediction SMALLINT NOT NULL CHECK (prediction in (0, 1)),
    probability DECIMAL NOT NULL,
    cm_value CHAR(2) NOT NULL,
    -- timestamp column
    created_at TIMESTAMP WITH TIME ZONE DEFAULT (CURRENT_TIMESTAMP AT TIME ZONE 'UTC') NOT NULL
);

--------------------------------------------------------------------------------
-- comments
--------------------------------------------------------------------------------

-- add table comment
COMMENT ON TABLE prediction IS 'stores training data to be ingested by reel-driver';

-- identifier column
COMMENT ON COLUMN prediction.imdb_id IS 'IMDB identifier for media item, and the primary key for this column';
-- value columns
COMMENT ON COLUMN prediction.prediction IS 'model prediction value, 1 for would_watch and 0 for would_not_watch';
COMMENT ON COLUMN prediction.probability IS 'probability of would_watch on 0 to 1 scale';
COMMENT ON COLUMN prediction.cm_value IS 'confusion matrix value, either ture positive, true negative, false postivie or false negative';
-- timestamps
COMMENT ON COLUMN prediction.created_at IS 'timestamp for initial database creation of item';

--------------------------------------------------------------------------------
-- end of instantiate_prediction.sql
--------------------------------------------------------------------------------
