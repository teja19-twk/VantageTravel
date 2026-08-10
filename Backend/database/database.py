import logging
from typing import Dict, Any, List, Optional
from supabase import create_client, Client
from .config import SUPABASE_URL, SUPABASE_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Initialize Supabase client safely
supabase: Optional[Client] = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Successfully initialized Supabase client.")
    except Exception as err:
        logger.warning(f"Could not connect to Supabase: {err}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY missing in config. Running in offline/fallback mode.")


def check_database_connection() -> bool:
    """Checks if the Supabase database connection is active."""
    if supabase is None:
        return False
    try:
        # Simple ping query to check connection
        supabase.table("predictions_log").select("id").limit(1).execute()
        return True
    except Exception as e:
        logger.error(f"Database connection ping failed: {e}")
        return False


def save_prediction(data: Dict[str, Any], table_name: str = "predictions_log") -> Optional[Dict[str, Any]]:
    """
    Saves a prediction record (Trip Budget, Visitor Crowd, or Climate) to Supabase.
    Gracefully handles missing tables (404/PGRST204/PGRST205 schema cache errors) without breaking.
    """
    if supabase is None:
        logger.info(f"[Offline] Prediction logged locally: {table_name}")
        return None

    try:
        response = supabase.table(table_name).insert(data).execute()
        return response.data if hasattr(response, "data") else None
    except Exception as e:
        # Catch missing table 404 / schema cache error quietly without flooding terminal
        logger.debug(f"[Supabase Log Info] Could not log to table '{table_name}': {e}")
        return None


def get_prediction_history(table_name: str = "predictions_log", limit: int = 50) -> List[Dict[str, Any]]:
    """
    Retrieves recent prediction history records from Supabase.
    """
    if supabase is None:
        return []

    try:
        response = supabase.table(table_name).select("*").order("created_at", desc=True).limit(limit).execute()
        return response.data if response.data else []
    except Exception as e:
        logger.error(f"Failed to fetch history from table '{table_name}': {e}")
        return []


def save_user_feedback(feedback_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Saves user feedback / rating into Supabase 'user_feedback' table.
    """
    return save_prediction(feedback_data, table_name="user_feedback")