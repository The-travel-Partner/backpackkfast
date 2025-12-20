import strawberry
from typing import List, Optional
from datetime import datetime
from authenticate.authentication import authenticate
from models import SimpleTrip as TripModel, Post as PostModel, Community as CommunityModel
from config import db, usercollection, apikey, SECRET_KEY, ALGORITHM, auth
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request


@strawberry.type
class User:
    """GraphQL User type matching authentication.User model"""
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    disabled: Optional[bool] = None


@strawberry.type  
class TripLocation:
    name: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_id: Optional[str] = None
    rating: Optional[float] = None
    place_type: Optional[str] = None


@strawberry.type
class TripDay:
    day: int
    weekday: Optional[str] = None
    places: List[TripLocation] = strawberry.field(default_factory=list)
    lunch_place: Optional[str] = None
    dinner_place: Optional[str] = None
    night_place: Optional[str] = None


@strawberry.type
class Trip:
    id: str
    user_id: str
    city_name: str
    no_of_days: int
    place_types: List[str] = strawberry.field(default_factory=list)
    itinerary: List[TripDay] = strawberry.field(default_factory=list)
    created_at: Optional[datetime] = None
    status: Optional[str] = None


@strawberry.type
class Community:
    id: str
    name: str
    description: Optional[str] = None
    creator_id: str
    member_count: Optional[int] = 0
    created_at: Optional[datetime] = None
    is_private: Optional[bool] = False


@strawberry.type
class Post:
    id: str
    user_id: str
    title: Optional[str] = None
    content: str
    community_id: Optional[str] = None
    likes: Optional[int] = 0
    comments_count: Optional[int] = 0
    created_at: Optional[datetime] = None


@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello from Backpackk GraphQL!"
    
    @strawberry.field
    def users(self) -> List[User]:
        # Minimal example - replace with actual user fetching logic
        return [
            User(
                email="john@example.com",
                first_name="John", 
                last_name="Doe",
                disabled=False
            ),
            User(
                email="jane@example.com",
                first_name="Jane", 
                last_name="Smith",
                disabled=False
            )
        ]
    
    @strawberry.field
    def trips(self) -> List[Trip]:
        # Sample data matching the actual MongoDB structure
        sample_places = [
            TripLocation(
                name="Wax Museum Udaipur",
                latitude=24.5953732,
                longitude=73.6872296,
                place_id="ChIJU11SpUvlZzkRlSL_l4aYtWw",
                rating=4.7,
                place_type="museum"
            ),
            TripLocation(
                name="Sajjangarh Monsoon Palace",
                latitude=24.5931853,
                longitude=73.6396056,
                place_id="ChIJw7A_ucr6ZzkRJTV3E4HVc4A",
                rating=4.4,
                place_type="tourist_attraction"
            )
        ]
        
        sample_days = [
            TripDay(
                day=1,
                weekday="Tuesday",
                places=sample_places,
                lunch_place="Bawarchi Restaurant",
                dinner_place="Natraj Dining Hall And Restaurant",
                night_place="Rootage Restaurant And Lounge"
            )
        ]
        
        return [
            Trip(
                id="trip1",
                user_id="1",
                city_name="Udaipur, Rajasthan, India",
                no_of_days=5,
                place_types=["museum", "tourist_attraction", "hindu_temple", "zoo", "night_club", "bar"],
                itinerary=sample_days,
                status="completed"
            )
        ]
    
    @strawberry.field
    def communities(self) -> List[Community]:
        # Minimal example - replace with actual community fetching logic
        return [
            Community(
                id="comm1",
                name="Travel Enthusiasts",
                description="Share your travel experiences",
                creator_id="1",
                member_count=150,
                is_private=False
            )
        ]
    
    @strawberry.field
    def posts(self, community_id: Optional[str] = None) -> List[Post]:
        # Minimal example - replace with actual post fetching logic
        return [
            Post(
                id="post1",
                user_id="1",
                title="Amazing trip to Japan",
                content="Just returned from an incredible journey...",
                community_id=community_id,
                likes=25,
                comments_count=5
            )
        ]


@strawberry.type
class Mutation:
    @strawberry.mutation
    def create_user(self, email: str, first_name: str, last_name: str) -> User:
        # Minimal example - replace with actual user creation logic
        return User(
            email=email,
            first_name=first_name,
            last_name=last_name,
            disabled=False
        )
    
    @strawberry.mutation
    def create_trip(
        self, 
        city_name: str,
        place_types: List[str],
        no_of_days: int
    ) -> Trip:
        # Minimal example - replace with actual trip creation logic
        return Trip(
            id="new_trip_id",
            user_id="current_user_id",
            city_name=city_name,
            no_of_days=no_of_days,
            place_types=place_types,
            itinerary=[],
            status="planning"
        )
    
    @strawberry.mutation
    def create_post(
        self, 
        content: str, 
        title: Optional[str] = None,
        community_id: Optional[str] = None
    ) -> Post:
        # Minimal example - replace with actual post creation logic  
        return Post(
            id="new_post_id",
            user_id="current_user_id",
            title=title,
            content=content,
            community_id=community_id,
            likes=0,
            comments_count=0
        )


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation)
