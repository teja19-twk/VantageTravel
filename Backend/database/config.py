import os
from pathlib import Path
from dotenv import load_dotenv

# Path configuration
BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)

# Database Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Base & Project Root Paths
BASE_DIR = BACKEND_DIR.parent
PROJECT_ROOT = BASE_DIR.parent

# Flexible Data Directory Resolution (supports root Datasets, APP Files/Datasets, and Backend/Datasets)
_data_candidates = [
    PROJECT_ROOT / "Datasets",
    BASE_DIR / "Datasets",
    BACKEND_DIR / "Datasets",
    Path.cwd() / "Datasets",
    Path.cwd().parent / "Datasets",
    PROJECT_ROOT / "datasets",
    BASE_DIR / "datasets",
    BACKEND_DIR / "datasets",
    Path.cwd() / "datasets",
    Path.cwd().parent / "datasets"
]

DATA_DIR = next((p for p in _data_candidates if p.exists()), PROJECT_ROOT / "Datasets")

# Flexible PKL directory resolution (supports 'Pkl' and 'pkl')
PKL_DIR = BASE_DIR / "Pkl" if (BASE_DIR / "Pkl").exists() else BASE_DIR / "pkl"

# Specific ML Model Directories with automatic folder resolution
BUDGET_PKL_DIR = PKL_DIR / "budget" if (PKL_DIR / "budget").exists() else PKL_DIR / "Budget"

_visitor_candidates = [PKL_DIR / "spot vistiors", PKL_DIR / "visitor", PKL_DIR / "Visitor"]
VISITOR_PKL_DIR = next((p for p in _visitor_candidates if p.exists()), PKL_DIR / "visitor")

_climate_candidates = [PKL_DIR / "Climate", PKL_DIR / "climate"]
CLIMATE_PKL_DIR = next((p for p in _climate_candidates if p.exists()), PKL_DIR / "climate")