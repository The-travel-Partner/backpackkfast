from pydantic import BaseModel
from typing import List

class tripgenModel(BaseModel):
    city_name: str
    place_types: list
    no_of_days: int
    travel_schedule: str
    weekdays: str


class getPhotos(BaseModel):
    photoref: list

class getPhotosByPlaceId(BaseModel):
    place_id: str