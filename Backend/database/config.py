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

# Workspace Paths
PROJECT_ROOT = BACKEND_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "Datasets"
PKL_DIR = BACKEND_DIR.parent / "Pkl"

# ML Artifact Directories
BUDGET_PKL_DIR = PKL_DIR / "budget"
VISITOR_PKL_DIR = PKL_DIR / "spot vistiors"
CLIMATE_PKL_DIR = PKL_DIR / "Climate"