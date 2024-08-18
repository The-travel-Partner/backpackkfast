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


from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bson.objectid import ObjectId

app = FastAPI()
mongostr = "mongodb+srv://admin:C5Qt4vWNogmRSlVi@backpackk.tmkdask.mongodb.net/"
client = AsyncIOMotorClient(mongostr)
db = client['backpackk']
users_collection = db['users']

def generate_verification_token(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def send_verification_email(user_email, token, user_id):
    sender_email = "nimishspslosal@gmail.com"
    sender_password = "axmdvfpmiewtzmsd"
    receiver_email = user_email

    subject = "Email Verification"
    body = f"Please verify your email by clicking on the following link: http://localhost:8000/verify/{user_id}/{token}"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = subject

    message.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_email, message.as_string())
        server.quit()
        print("Verification email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")



class UserRegistration(BaseModel):
    email: EmailStr
    password: str

@app.post("/register/")
async def register_user(user: UserRegistration, background_tasks: BackgroundTasks):
    existing_user = await users_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists.")

    token = generate_verification_token()
    user_data = {
        "email": user.email,
        "password": user.password,
        "verified": False,
        "verification_token": token
    }
    new_user = await users_collection.insert_one(user_data)
    user_id = str(new_user.inserted_id)

    background_tasks.add_task(send_verification_email, user.email, token, user_id)

    return {"message": "User registered successfully. Please check your email to verify your account."}

@app.get("/verify/{user_id}/{token}")
async def verify_user(user_id: str, token: str):
    user = await users_collection.find_one({"_id": ObjectId(user_id), "verification_token": token})
    if user:
        await users_collection.update_one({"_id": ObjectId(user_id)}, {"$set": {"verified": True}, "$unset": {"verification_token": ""}})
        return {"message": "Email verified successfully!"}
    else:
        raise HTTPException(status_code=400, detail="Verification failed: Invalid token or user ID.")

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
