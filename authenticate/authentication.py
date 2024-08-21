from fastapi import FastAPI, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from motor.motor_asyncio import AsyncIOMotorClient

class authenticate:
    def __init__(self, secretkey, algorithm, usercollection):
        self.SECRET_KEY: str = secretkey
        self.ALGORITHM: str = algorithm
        self.users_collection: AsyncIOMotorClient = usercollection

    class Token(BaseModel):
        access_token: str
        token_type: str
        first_name: str


    class TokenData(BaseModel):
        email: Optional[EmailStr] = None

    class UserCreate(BaseModel):
        email: EmailStr
        first_name: str
        last_name: str
        password: str


    class User(BaseModel):
        email: EmailStr
        first_name: Optional[str] = None
        last_name: Optional[str] = None
        disabled: Optional[bool] = None

    class UserInDB(User):
        hashed_password: str


    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

    def verify_password(self,plain_password, hashed_password):
        return self.pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self,password):
        return self.pwd_context.hash(password)

    async def get_user_by_email(self, email: str):
        user = await self.users_collection.find_one({"email": email})
        if user:
            return self.UserInDB(**user)

    async def authenticate_user(self,email: str, password: str):
        user = await self.get_user_by_email(email)
        if not user:
            return False
        if not self.verify_password(password, user.hashed_password):
            return False
        return user

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)
        return encoded_jwt

    async def get_current_user(self, token: str = Depends(OAuth2PasswordBearer(tokenUrl="token"))):
        credential_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                raise credential_exception
            token_data = self.TokenData(email=email)
        except JWTError:
            raise credential_exception
        user = await self.get_user_by_email(token_data.email)
        if user is None:
            raise credential_exception
        return user

    async def get_current_active_user(self, current_user: UserInDB = Depends(get_current_user)):
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user
