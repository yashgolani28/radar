CREATE TABLE public.radar_data (
    id SERIAL PRIMARY KEY,
    timestamp DOUBLE PRECISION,
    datetime TEXT,
    sensor TEXT,
    object_id TEXT,
    type TEXT,
    confidence DOUBLE PRECISION,
    speed_kmh DOUBLE PRECISION,
    velocity DOUBLE PRECISION,
    distance DOUBLE PRECISION,
    direction TEXT,
    signal_level DOUBLE PRECISION,
    doppler_frequency DOUBLE PRECISION,
    snapshot_path TEXT,
    reviewed INTEGER DEFAULT 0,
    flagged INTEGER DEFAULT 0,
    radar_distance DOUBLE PRECISION,
    visual_distance DOUBLE PRECISION
);

CREATE TABLE public.users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('admin', 'viewer')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE public.user_activity (
    user_id INTEGER PRIMARY KEY,
    last_activity TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES public.users(id) ON DELETE CASCADE
);
