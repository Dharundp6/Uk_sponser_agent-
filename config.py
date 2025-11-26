"""
Configuration settings for UK Visa Sponsorship Assistant
Contains cache paths and system-wide constants
"""
from pathlib import Path

# Cache Directory Configuration
CACHE_DIR = Path.home() / ".uk_visa_cache"
FAISS_INDEX_FILE = CACHE_DIR / "faiss_index.bin"
EMBEDDINGS_CACHE_FILE = CACHE_DIR / "embeddings_cache.pkl"
DATA_CACHE_FILE = CACHE_DIR / "sponsors_data.pkl"
METADATA_CACHE_FILE = CACHE_DIR / "metadata_cache.pkl"

# Create cache directory if it doesn't exist
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Embedding Configuration
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
EMBEDDING_DIMENSION = 384
MAX_SEQUENCE_LENGTH = 256

# Model Configuration
GEMINI_MODEL_NAME = "gemini-flash-latest"

# Search Configuration
USE_LLM_FILTERING = True  # Set to False to disable LLM filtering (faster, but less accurate)

# Visa Eligibility Constants
MIN_SALARY = 38700
NEW_ENTRANT_SALARY = 30960


def clear_cache():
    """Clear all cached data to force fresh database creation"""
    cache_files = [
        FAISS_INDEX_FILE,
        EMBEDDINGS_CACHE_FILE,
        DATA_CACHE_FILE,
        METADATA_CACHE_FILE
    ]

    cleared = []
    for cache_file in cache_files:
        if cache_file.exists():
            try:
                cache_file.unlink()
                cleared.append(str(cache_file.name))
            except Exception as e:
                print(f"⚠️  Could not delete {cache_file}: {e}")

    if cleared:
        print(f"🗑️  Cleared cache files: {', '.join(cleared)}")
    else:
        print("📂 No cache files found to clear")

    return cleared
