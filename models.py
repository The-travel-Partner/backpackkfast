"""
Pydantic models for MongoDB collections with BSON ObjectId handling.
Includes models for Trip, Post, and Community collections.
User models are imported from authenticate.authentication
"""

from pydantic import BaseModel, Field, EmailStr, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId

# Import User models from authentication
from authenticate.authentication import authenticate


class PyObjectId(ObjectId):
    """Custom ObjectId class for Pydantic validation."""
    
    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type, _handler
    ):
        from pydantic_core import core_schema
        return core_schema.json_or_python_schema(
            json_schema=core_schema.str_schema(),
            python_schema=core_schema.union_schema([
                core_schema.is_instance_schema(ObjectId),
                core_schema.chain_schema([
                    core_schema.str_schema(),
                    core_schema.no_info_plain_validator_function(cls.validate),
                ])
            ]),
            serialization=core_schema.plain_serializer_function_ser_schema(
                str,
                return_schema=core_schema.str_schema(),
            ),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if ObjectId.is_valid(v):
                return ObjectId(v)
        raise ValueError("Invalid ObjectId")


# Base configuration for all models
class BaseConfig:
    # Allow population by field name (updated for Pydantic v2)
    populate_by_name = True
    # JSON encoders for ObjectId
    json_encoders = {ObjectId: str}
    # Use enum values
    use_enum_values = True


# --- TRIP MODELS (Based on actual MongoDB structure) ---

class PlaceDetails(BaseModel):
    """Individual place details structure."""
    name: Dict[str, str] = {}  # {0: "Place Name", 1: "Another Place"}
    lat: Dict[str, float] = {}
    lng: Dict[str, float] = {}
    rating: Dict[str, float] = {}
    number: Dict[str, int] = {}  # Number of reviews
    place_id: Dict[str, str] = {}
    type: Dict[str, str] = {}
    weight_avg: Dict[str, float] = {}
    distance: Dict[str, int] = {}
    opening_hours: Dict[str, List[str]] = {}
    
    class Config(BaseConfig):
        pass


class MealPlace(BaseModel):
    """Meal place details (Lunch/Dinner/Night)."""
    Name: Dict[str, str] = {}
    lat: Dict[str, float] = {}
    lng: Dict[str, float] = {}
    rating: Dict[str, float] = {}
    number: Dict[str, int] = {}
    place_id: Dict[str, str] = {}
    type: Dict[str, str] = {}
    weight_avg: Dict[str, float] = {}
    distance: Dict[str, int] = {}
    opening_hours: Dict[str, List[str]] = {}
    
    class Config(BaseConfig):
        pass


class TimeSlots(BaseModel):
    """Time slots for activities."""
    Start: Dict[str, str] = {}  # {0: "09:00", 1: "10:59", "Lunch": "13:00"}
    End: Dict[str, str] = {}
    Slot_Duration: Dict[str, Any] = Field(alias="Slot Duration", default={})  # Mix of strings and ints
    Travel_Duration: Dict[str, Optional[int]] = Field(alias="Travel Duration", default={})
    
    class Config(BaseConfig):
        pass


class DayItinerary(PlaceDetails):
    """Daily itinerary extending place details."""
    Weekday: str
    Lunch: Optional[MealPlace] = None
    Dinner: Optional[MealPlace] = None
    Night: Optional[MealPlace] = None
    Slots: Optional[TimeSlots] = None
    
    class Config(BaseConfig):
        pass


class TripData(BaseModel):
    """Complete trip data structure."""
    city_name: str
    place_types: List[str] = []
    no_of_days: int
    trip: Dict[str, DayItinerary] = {}  # {"Day 1": DayItinerary, "Day 2": ...}
    
    class Config(BaseConfig):
        pass


class TripInDB(BaseModel):
    """Trip model as stored in MongoDB - matches actual structure."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    status: Optional[str] = "completed"
    # The actual trip data
    city_name: str
    place_types: List[str] = []
    no_of_days: int
    trip: Dict[str, DayItinerary] = {}
    
    class Config(BaseConfig):
        pass


class TripCreate(BaseModel):
    """Model for creating a new trip."""
    city_name: str
    place_types: List[str]
    no_of_days: int
    
    class Config(BaseConfig):
        pass


class TripResponse(BaseModel):
    """Public trip model for API responses."""
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    city_name: str
    place_types: List[str] = []
    no_of_days: int
    trip: Dict[str, DayItinerary] = {}
    created_at: Optional[datetime] = None
    status: Optional[str] = None
    
    class Config(BaseConfig):
        pass


# Simplified models for GraphQL (to avoid complexity)
class SimpleTripLocation(BaseModel):
    """Simplified location for GraphQL."""
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_id: Optional[str] = None
    rating: Optional[float] = None
    place_type: Optional[str] = None
    
    class Config(BaseConfig):
        pass


class SimpleTripDay(BaseModel):
    """Simplified day model for GraphQL."""
    day: int
    weekday: Optional[str] = None
    places: List[SimpleTripLocation] = []
    lunch_place: Optional[str] = None
    dinner_place: Optional[str] = None
    night_place: Optional[str] = None
    
    class Config(BaseConfig):
        pass


class SimpleTrip(BaseModel):
    """Simplified trip model for GraphQL responses."""
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    city_name: str
    no_of_days: int
    place_types: List[str] = []
    itinerary: List[SimpleTripDay] = []
    created_at: Optional[datetime] = None
    status: Optional[str] = None
    
    class Config(BaseConfig):
        pass


# --- COMMUNITY MODELS ---

class CommunityBase(BaseModel):
    """Base community model."""
    name: str
    description: Optional[str] = None
    is_private: Optional[bool] = False
    category: Optional[str] = None
    rules: Optional[List[str]] = []
    
    class Config(BaseConfig):
        pass


class CommunityCreate(CommunityBase):
    """Model for community creation."""
    pass


class CommunityUpdate(BaseModel):
    """Model for community updates."""
    name: Optional[str] = None
    description: Optional[str] = None
    is_private: Optional[bool] = None
    category: Optional[str] = None
    rules: Optional[List[str]] = None
    
    class Config(BaseConfig):
        pass


class CommunityInDB(CommunityBase):
    """Community model as stored in database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    creator_id: PyObjectId
    moderators: Optional[List[PyObjectId]] = []
    members: Optional[List[PyObjectId]] = []
    member_count: Optional[int] = 0
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    is_active: Optional[bool] = True
    
    class Config(BaseConfig):
        pass


class Community(CommunityBase):
    """Public community model."""
    id: PyObjectId = Field(alias="_id")
    creator_id: PyObjectId
    member_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    is_active: Optional[bool] = True
    
    class Config(BaseConfig):
        pass


# --- POST MODELS ---

class PostBase(BaseModel):
    """Base post model."""
    title: Optional[str] = None
    content: str
    post_type: Optional[str] = "text"  # text, image, link, trip_share
    is_public: Optional[bool] = True
    tags: Optional[List[str]] = []
    
    class Config(BaseConfig):
        pass


class PostCreate(PostBase):
    """Model for post creation."""
    community_id: Optional[PyObjectId] = None
    trip_id: Optional[PyObjectId] = None  # For trip sharing posts
    images: Optional[List[str]] = []  # Image URLs


class PostUpdate(BaseModel):
    """Model for post updates."""
    title: Optional[str] = None
    content: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None
    
    class Config(BaseConfig):
        pass


class CommentBase(BaseModel):
    """Base comment model."""
    content: str
    
    class Config(BaseConfig):
        pass


class CommentCreate(CommentBase):
    """Model for comment creation."""
    parent_comment_id: Optional[PyObjectId] = None  # For nested comments


class CommentInDB(CommentBase):
    """Comment model as stored in database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    post_id: PyObjectId
    user_id: PyObjectId
    parent_comment_id: Optional[PyObjectId] = None
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    likes: Optional[int] = 0
    is_deleted: Optional[bool] = False
    
    class Config(BaseConfig):
        pass


class Comment(CommentBase):
    """Public comment model."""
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    parent_comment_id: Optional[PyObjectId] = None
    created_at: Optional[datetime] = None
    likes: Optional[int] = 0
    
    class Config(BaseConfig):
        pass


class PostInDB(PostBase):
    """Post model as stored in database."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    community_id: Optional[PyObjectId] = None
    trip_id: Optional[PyObjectId] = None
    images: Optional[List[str]] = []
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    likes: Optional[int] = 0
    comments_count: Optional[int] = 0
    shares: Optional[int] = 0
    is_deleted: Optional[bool] = False
    
    class Config(BaseConfig):
        pass


class Post(PostBase):
    """Public post model with related data."""
    id: PyObjectId = Field(alias="_id")
    user_id: PyObjectId
    community_id: Optional[PyObjectId] = None
    trip_id: Optional[PyObjectId] = None
    images: Optional[List[str]] = []
    created_at: Optional[datetime] = None
    likes: Optional[int] = 0
    comments_count: Optional[int] = 0
    shares: Optional[int] = 0
    
    class Config(BaseConfig):
        pass


# --- UTILITY MODELS ---

class PaginatedResponse(BaseModel):
    """Generic paginated response model."""
    items: List[Any]
    total: int
    page: int
    per_page: int
    pages: int
    has_next: bool
    has_prev: bool
    
    class Config(BaseConfig):
        pass


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=10, ge=1, le=100)
    
    class Config(BaseConfig):
        pass


# --- TRIP GENERATION RESPONSE MODEL (From Scratch) ---

class TripSlots(BaseModel):
    """Time slots for trip activities."""
    Start: Dict[str, str] = {}  # {"0": "09:00", "1": "10:59", "Lunch": "13:00"}
    End: Dict[str, str] = {}    # {"0": "10:46", "1": "12:46", "Lunch": "14:30"}
    Slot_Duration: Dict[str, Any] = Field(alias="Slot Duration", default={})  # Mix of strings and ints
    Travel_Duration: Dict[str, Optional[int]] = Field(alias="Travel Duration", default={})  # Can be null
    
    class Config(BaseConfig):
        populate_by_name = True


class TripMealPlace(BaseModel):
    """Meal place details for Lunch/Dinner/Night."""
    Name: Dict[str, str] = {}           # {"0": "Restaurant Name"}
    lat: Dict[str, float] = {}          # {"0": 24.5849393}
    lng: Dict[str, float] = {}          # {"0": 73.6952694}
    rating: Dict[str, float] = {}       # {"0": 4.6}
    number: Dict[str, int] = {}         # {"0": 18571}
    place_id: Dict[str, str] = {}       # {"0": "ChIJGelIePzlZzkRrAXY9075AK8"}
    type: Dict[str, str] = {}           # {"0": "vegetarian_restaurant"}
    weight_avg: Dict[str, float] = {}   # {"0": 13001.08}
    distance: Dict[str, int] = {}       # {"0": 410365}
    opening_hours: Dict[str, List[str]] = {}  # {"0": ["Monday: 9:00 AM – 10:20 PM", ...]}
    
    class Config(BaseConfig):
        pass


class TripDayData(BaseModel):
    """Complete day itinerary data."""
    Name: Dict[str, str] = {}                    # {"0": "Place Name", "1": "Another Place"}
    lat: Dict[str, float] = {}                   # {"0": 24.5953732, "1": 24.5931853}
    lng: Dict[str, float] = {}                   # {"0": 73.6872296, "1": 73.6396056}
    rating: Dict[str, float] = {}                # {"0": 4.7, "1": 4.4}
    number: Dict[str, int] = {}                  # {"0": 42598, "1": 30244}
    place_id: Dict[str, str] = {}                # {"0": "ChIJU11SpUvlZzkRlSL_l4aYtWw"}
    type: Dict[str, str] = {}                    # {"0": "museum", "1": "tourist_attraction"}
    weight_avg: Dict[str, float] = {}            # {"0": 29820.01, "1": 21172.12}
    distance: Dict[str, int] = {}                # {"0": 409007, "1": 417988}
    opening_hours: Dict[str, List[str]] = {}     # {"0": ["Monday: 9:00 AM – 9:00 PM", ...]}
    Weekday: str                                 # "Tuesday"
    Lunch: Optional[TripMealPlace] = None        # Lunch place details
    Dinner: Optional[TripMealPlace] = None       # Dinner place details
    Night: Optional[TripMealPlace] = None        # Night place details
    Slots: Optional[TripSlots] = None            # Time slots
    
    class Config(BaseConfig):
        pass


class TripGenerationData(BaseModel):
    """Model for the complete trip generation response matching the provided JSON structure."""
    city_name: str
    place_types: List[str]
    no_of_days: int
    trip: Dict[str, TripDayData]  # {"Day 1": TripDayData, "Day 2": TripDayData, ...}
    
    class Config(BaseConfig):
        pass


