"""
Pydantic models for flight search requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Optional


class FlightSearchRequest(BaseModel):
    """Request model for flight search via SerpAPI Google Flights."""
    
    departure_id: str = Field(
        ..., 
        description="Departure airport code (e.g., 'JFK', 'DEL') or location kgmid"
    )
    arrival_id: str = Field(
        ..., 
        description="Arrival airport code (e.g., 'LAX', 'BOM') or location kgmid"
    )
    outbound_date: str = Field(
        ..., 
        description="Outbound date in YYYY-MM-DD format (e.g., '2025-12-01')"
    )
    return_date: Optional[str] = Field(
        None, 
        description="Return date in YYYY-MM-DD format (required for round trip)"
    )
    type: int = Field(
        default=1,
        description="Flight type: 1=Round trip, 2=One way, 3=Multi-city",
        ge=1,
        le=3
    )
    travel_class: int = Field(
        default=1,
        description="Travel class: 1=Economy, 2=Premium economy, 3=Business, 4=First",
        ge=1,
        le=4
    )
    adults: int = Field(
        default=1,
        description="Number of adult passengers",
        ge=1,
        le=9
    )
    children: int = Field(
        default=0,
        description="Number of children",
        ge=0,
        le=9
    )
    infants_in_seat: int = Field(
        default=0,
        description="Number of infants in seat",
        ge=0,
        le=9
    )
    infants_on_lap: int = Field(
        default=0,
        description="Number of infants on lap",
        ge=0,
        le=9
    )
    currency: str = Field(
        default="USD",
        description="Currency code (e.g., 'USD', 'EUR', 'INR')"
    )
    gl: str = Field(
        default="us",
        description="Country code (e.g., 'us', 'uk', 'in')"
    )
    hl: str = Field(
        default="en",
        description="Language code (e.g., 'en', 'es', 'fr')"
    )
    sort_by: int = Field(
        default=1,
        description="Sort order: 1=Top flights, 2=Price, 3=Departure time, 4=Arrival time, 5=Duration, 6=Emissions",
        ge=1,
        le=6
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "departure_id": "DEL",
                "arrival_id": "BOM",
                "outbound_date": "2025-12-01",
                "return_date": "2025-12-10",
                "type": 1,
                "travel_class": 1,
                "adults": 2,
                "children": 0,
                "currency": "INR",
                "gl": "in",
                "hl": "en"
            }
        }
