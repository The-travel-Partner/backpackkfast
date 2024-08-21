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

from fastapi.responses import JSONResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware


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



@app.post("/register")
async def register_user(user: auth.UserCreate, background_tasks: BackgroundTasks):
    existing_user = await usercollection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists.")
    
    token = generate_verification_token()
    hashed_password = auth.get_password_hash(user.password)
    user_dict= user.model_dump()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    user_dict["disabled"] = False
    user_dict["verified"] = False
    user_dict["verification_token"] = token

    new_user = await usercollection.insert_one(user_dict)
    user_id = str(new_user.inserted_id)

    background_tasks.add_task(send_verification_email, user.email, token, user_id)

    return {"message": "User registered successfully. Please check your email to verify your account."}

@app.get("/verify/{user_id}/{token}")
async def verify_user(user_id: str, token: str):
    user = await usercollection.find_one({"_id": ObjectId(user_id), "verification_token": token})
    if user:
        await usercollection.update_one({"_id": ObjectId(user_id)}, {"$set": {"verified": True}, "$unset": {"verification_token": ""}})
        return {"message": "Email verified successfully!"}
    else:
        raise HTTPException(status_code=400, detail="Verification failed: Invalid token or user ID.")

@app.get("/users/me", response_model=auth.User)
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
import os

async def find_or_create_user(email, first_name, last_name):
    user = await usercollection.find_one({"email": email})
    if user:
        return user

    
    new_user = UserCreate(
        email=email,
        first_name=first_name,
        last_name=last_name,
        password=''  
    ).dict()
    new_user["verified"] = True  
    new_user["disabled"] = False
    await usercollection.insert_one(new_user)
    return new_user

secret_key = os.getenv("SESSION_SECRET_KEY", "default_fallback_secret_key")
from jose import jwt
import requests
from fastapi.responses import RedirectResponse
from fastapi import Request
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
oauth = OAuth()
oauth.register(
    name='google',
    client_id='794713488480-8iqh9m6p3a93clvqrfrdjakt8q22movg.apps.googleusercontent.com',
    client_secret='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',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    token_url='https://accounts.google.com/o/oauth2/token',
    redirect_uri='http://localhost:8000/auth/callback',
    client_kwargs={'scope': 'openid profile email'},
)
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from starlette.requests import Request


google_router = APIRouter()
app.include_router(google_router, prefix="/auth")

SECRET_KEY = os.getenv("SECRET_KEY", "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")

@app.get("/google")
async def google_auth(request: Request):
    # Construct the redirect URI
    redirect_uri = request.url_for("google_callback")

    # Ensure that the redirect_uri matches what is configured in Google Cloud Console
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/callback")
async def google_callback(request: Request):
    try:
        # Retrieve the token from Google
        token = await oauth.google.authorize_access_token(request)

        # Retrieve user info from Google
        user_info = await oauth.google.get('userinfo', token=token)
        email = user_info.data['email']
        first_name = user_info.data['given_name']
        last_name = user_info.data['family_name']

        # Find or create the user in your database
        user = await find_or_create_user(email, first_name, last_name)

        # Generate a session token using JWT
        session_token = jwt.encode({"sub": str(user["_id"])}, SECRET_KEY, algorithm="HS256")

        # Redirect to the home page with the session token in a cookie
        response = RedirectResponse(url="/")
        response.set_cookie(key="session_token", value=session_token, httponly=True)

        return response

    except Exception as e:
        # Raise an HTTPException with detailed error information
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")




