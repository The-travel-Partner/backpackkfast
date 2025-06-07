from pandas import DataFrame
from pydantic import BaseModel

class bestPlacesModel(BaseModel):
    number_of_days : int
    modelData : dict
