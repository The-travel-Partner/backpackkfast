from pydantic import BaseModel, EmailStr



class contactModel(BaseModel):
    email: EmailStr
    first_name:str
    last_name:str
    message:str