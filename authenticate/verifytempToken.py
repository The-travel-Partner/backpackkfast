from pydantic import BaseModel


class VerifyToken(BaseModel):
    temptoken: str