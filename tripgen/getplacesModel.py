from pydantic import BaseModel


class getplacesModel(BaseModel):
    city_name : str
    place_types: list