from pydantic import BaseModel, Field
from typing import Optional, List, Any


# =====================================================================
# INPUT SCHEMAS (API Requests)
# =====================================================================

class TripBudgetInput(BaseModel):
    duration_days: int = Field(..., ge=1, description="Duration of the trip in days")
    num_travelers: int = Field(..., ge=1, description="Number of travelers")
    route_distance_km: float = Field(..., ge=0.0, description="Total route distance in kilometers")
    transport_mode: str = Field("car", description="Mode of transport: car, bike, bus, train, flight")
    accommodation_tier: str = Field("mid-range", description="Tier: budget, mid-range, luxury")
    season: str = Field("Winter", description="Season: Winter, Summer, Monsoon, Post-Monsoon")
    stay_cost_est: Optional[float] = Field(0.0, description="Optional stay cost estimate")
    food_cost_est: Optional[float] = Field(0.0, description="Optional food cost estimate")
    entry_fees_est: Optional[float] = Field(0.0, description="Optional entry fees estimate")
    tolls_and_parking_est: Optional[float] = Field(0.0, description="Optional tolls and parking estimate")


class VisitorInput(BaseModel):
    place_name: str = Field(..., description="Name of the tourist spot")
    district: str = Field(..., description="District where the spot is located")
    month: int = Field(..., ge=1, le=12, description="Month of visit (1-12)")
    season: str = Field("Winter", description="Season: Winter, Summer, Monsoon, Post-Monsoon")


class ClimateInput(BaseModel):
    district: str = Field(..., description="District name")
    month: int = Field(..., ge=1, le=12, description="Month of visit (1-12)")
    temperature: Optional[float] = Field(0.0, description="Current or expected temperature in Celsius")
    humidity: Optional[float] = Field(0.0, description="Humidity percentage")
    rainfall: Optional[float] = Field(0.0, description="Rainfall in mm")


class RecommendationQuery(BaseModel):
    selected_spot: Optional[str] = Field(None, description="Reference spot for similarity recommendations")
    district: Optional[str] = Field(None, description="Target district filter")
    category: Optional[str] = Field("all", description="Category: heritage, nature, spiritual, all")
    season: Optional[str] = Field(None, description="Preferred season")
    budget: Optional[float] = Field(0.0, description="Maximum budget")
    crowd: Optional[int] = Field(0, description="Max acceptable crowd limit")
    transport: Optional[str] = Field("car", description="Transport mode")


class UserFeedbackInput(BaseModel):
    user_name: Optional[str] = Field("Anonymous", description="User name")
    rating: int = Field(..., ge=1, le=5, description="Star rating from 1 to 5")
    feedback_text: str = Field(..., description="Feedback comments")
    spot_name: Optional[str] = Field(None, description="Spot name reviewed")


# =====================================================================
# RESPONSE SCHEMAS (API Outputs)
# =====================================================================

class TripBudgetResponse(BaseModel):
    duration_days: int
    num_travelers: int
    transport_mode: str
    accommodation_tier: str
    stay_cost: float
    food_cost: float
    transport_cost: float
    entry_fees: float
    estimated_total_budget: float
    per_person_cost: float
    status: str = "success"


class VisitorResponse(BaseModel):
    place_name: str
    district: str
    month: int
    season: str
    predicted_visitors: int
    crowd_density: str
    recommended_time: str
    status: str = "success"


class ClimateResponse(BaseModel):
    district: str
    month: int
    predicted_max_temp_c: float
    predicted_min_temp_c: float
    rain_probability_percent: float
    weather_condition: str
    travel_advisory: str
    status: str = "success"


class SpotRecommendation(BaseModel):
    spot_name: str
    district: str
    category: str
    rating: float
    season: str
    expected_crowd: int
    estimated_budget: float
    transport: str
    lat: float
    lon: float
    entry_fee: float
    reviews: int


class APIResponse(BaseModel):
    status: str
    message: str
    data: Optional[Any] = None