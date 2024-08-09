from urllib.request import Request

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from tripgen.tripgenModel import tripgenModel
import tripgen.tripgenModel
from tripgen.tripcreator import TripCreator
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta

mongostr = "mongodb+srv://admin:C5Qt4vWNogmRSlVi@backpackk.tmkdask.mongodb.net/"
client = AsyncIOMotorClient(mongostr)
db = client['backpackk']
usercollection = db['users']
from authenticate.authentication import authenticate

SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
auth = authenticate(secretkey=SECRET_KEY, algorithm=ALGORITHM, usercollection=usercollection)

app = FastAPI()

async def current_active_user_dependency(current_user: auth.UserInDB = Depends(auth.get_current_user)):
    return await auth.get_current_active_user(current_user)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/token", response_model=auth.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await auth.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register", response_model=auth.User)
async def register_user(user: auth.UserCreate):
    existing_user = await auth.get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    hashed_password = auth.get_password_hash(user.password)
    user_dict = user.model_dump()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    user_dict["disabled"] = False
    await usercollection.insert_one(user_dict)
    return auth.User(email=user.email, first_name=user.first_name, last_name=user.last_name)

@app.get("/users/me/", response_model=auth.User)
async def read_users_me(current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    return current_user
@app.post('/tripgenerator')
async def generator(param: tripgenModel):
    city_name = param.city_name
    place_types = param.place_types
    no_of_days = param.no_of_days
    trip = TripCreator(city_name=city_name, place_types=place_types, no_of_days=no_of_days)
    new_trip = await trip.create_trip()
    return new_trip
