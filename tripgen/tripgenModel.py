from pydantic import BaseModel


class tripgenModel(BaseModel):
    city_name: str
    place_types: list
    no_of_days: int
