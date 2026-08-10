from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any

from database.schemas import (
    TripBudgetInput,
    VisitorInput,
    ClimateInput,
    RecommendationQuery,
    UserFeedbackInput,
    APIResponse,
)
from database.database import (
    check_database_connection,
    get_prediction_history,
    save_user_feedback,
)
from predict import (
    predict_trip_budget,
    predict_visitors,
    predict_climate,
)
from recommendation import (
    get_all_districts,
    get_spots,
    get_nearby_amenities,
    recommend_places,
)

app = FastAPI(
    title="VantageTravel ML Platform API",
    description="Machine Learning Backend for Trip Budgeting, Visitor Crowd Forecasting, Climate Analysis & Destination Recommendations.",
    version="1.0.0",
)

# Enable CORS for Frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """Health check endpoint."""
    db_status = check_database_connection()
    return {
        "status": "online",
        "app_name": "VantageTravel ML Platform",
        "database_connected": db_status,
    }


# =====================================================================
# ML PREDICTION ENDPOINTS
# =====================================================================

@app.post("/api/predict/budget", response_model=Dict[str, Any])
def api_predict_budget(data: TripBudgetInput):
    """Predicts detailed trip costs (stay, food, transport, entry fees, total budget)."""
    result = predict_trip_budget(data)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@app.post("/api/predict/visitors", response_model=Dict[str, Any])
def api_predict_visitors(data: VisitorInput):
    """Predicts tourist crowd volume and density level."""
    result = predict_visitors(data)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


@app.post("/api/predict/climate", response_model=Dict[str, Any])
def api_predict_climate(data: ClimateInput):
    """Predicts district weather forecast and rain probability."""
    result = predict_climate(data)
    if result.get("status") == "error":
        raise HTTPException(status_code=500, detail=result.get("message"))
    return result


# =====================================================================
# SPOT RECOMMENDATIONS & AMENITIES ENDPOINTS
# =====================================================================

@app.get("/api/districts")
def api_get_districts():
    """Returns unique list of supported districts."""
    return {"districts": get_all_districts()}


@app.get("/api/spots")
def api_get_spots(
    district: Optional[str] = Query(None, description="Filter by district"),
    category: Optional[str] = Query(None, description="Filter by category"),
):
    """Fetches tourist spots filtered by district and category."""
    spots = get_spots(district=district, category=category)
    return {"spots": spots, "total": len(spots)}


@app.get("/api/amenities")
def api_get_amenities(
    spot_name: str = Query(..., description="Tourist spot name"),
    district: str = Query("Hyderabad", description="District name"),
    lat: float = Query(17.3850, description="Latitude"),
    lon: float = Query(78.4867, description="Longitude"),
):
    """Returns nearby hotels, restaurants, ATMs, hospitals, parking, petrol pumps, and restrooms."""
    try:
        if not spot_name or not spot_name.strip():
            raise HTTPException(
                status_code=400,
                detail="spot_name is required"
            )

        amenities = get_nearby_amenities(spot_name=spot_name, district=district, lat=lat, lon=lon)
        return {
            "spot_name": spot_name,
            "district": district,
            "latitude": lat,
            "longitude": lon,
            "amenities": amenities,
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Amenities Error] {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch amenities: {str(e)}"
        )


@app.post("/api/recommendations")
def api_recommend_places(query: RecommendationQuery):
    """Recommends top spots matching selected reference spot, budget, season, and crowd preferences."""
    recommendations = recommend_places(
        selected_spot=query.selected_spot,
        district=query.district,
        category=query.category,
        season=query.season,
        budget=query.budget or 0.0,
        crowd=query.crowd or 0,
        transport=query.transport or "car",
    )
    return {"recommendations": recommendations, "total": len(recommendations)}


# =====================================================================
# DATABASE & HISTORY ENDPOINTS
# =====================================================================

@app.get("/api/history")
def api_get_history(table_name: str = "trip_budget_log", limit: int = 50):
    """Fetches prediction history log from Supabase."""
    history = get_prediction_history(table_name=table_name, limit=limit)
    return {"table": table_name, "records": history}


@app.post("/api/feedback")
def api_save_feedback(feedback: UserFeedbackInput):
    """Saves user rating and feedback review to Supabase."""
    res = save_user_feedback(feedback.dict())
    return {"status": "success", "saved": res}
