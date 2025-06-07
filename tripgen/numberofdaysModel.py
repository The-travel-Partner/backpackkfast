from pydantic import BaseModel


class numberofdays(BaseModel):
    travel_schedule: str
    weekdays: str