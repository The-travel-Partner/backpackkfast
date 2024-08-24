from pydantic import BaseModel
class resetpass(BaseModel):
    token:str
    new_password:str
