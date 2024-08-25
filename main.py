from urllib.request import Request

import asyncio
import requests
from jose import jwt
from fastapi import FastAPI, Depends, HTTPException, status, Query,Request
from fastapi import FastAPI, Response
from fastapi.security import OAuth2PasswordRequestForm, OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from tripgen.tripgenModel import tripgenModel
import tripgen.tripgenModel
from tripgen.tripcreator import TripCreator
from passlib.context import CryptContext
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime, timedelta
from authenticate.verifytempToken import VerifyToken
from fastapi.responses import JSONResponse
from authlib.integrations.starlette_client import OAuth, OAuthError
from starlette.middleware.sessions import SessionMiddleware
from io import BytesIO
from PIL import Image
from motor.motor_asyncio import AsyncIOMotorClient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, EmailStr
import random
import string
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from bson.objectid import ObjectId

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
    existing_user = await usercollection.find_one({"email": user.email})
    name = existing_user.get('first_name')
    print(name)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer", "first_name": name}


def generate_verification_token(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


def send_verification_email(user_email, token, user_id):
    sender_email = "nimishspslosal@gmail.com"
    sender_password = "axmdvfpmiewtzmsd"
    receiver_email = user_email

    subject = "Email Verification"
    body = f"Please verify your email by clicking on the following link: https://backpackkfast-fcvonqkgya-el.a.run.app/verify/{user_id}/{token}"

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
    user_dict = user.model_dump()
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
        await usercollection.update_one({"_id": ObjectId(user_id)},
                                        {"$set": {"verified": True}, "$unset": {"verification_token": ""}})
        return {"message": "Email verified successfully!"}
    else:
        raise HTTPException(status_code=400, detail="Verification failed: Invalid token or user ID.")


@app.get("/users/me", response_model=auth.User)
async def read_users_me(current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    return current_user


@app.post('/tripgenerator')
async def generator(param: tripgenModel, request: Request, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        city_name = param.city_name
        place_types = param.place_types
        no_of_days = param.no_of_days
        trip = TripCreator(request=request,city_name=city_name, place_types=place_types, no_of_days=no_of_days)
        new_trip = await trip.create_trip()

        collection = str(current_user.email)
        usertripCollection = db[collection]
        await usertripCollection.insert_one({'city_name': city_name, 'place_types': place_types, 'no_of_days': no_of_days, 'trip': new_trip})
        return new_trip


import os


async def find_or_create_user(email, first_name, last_name='nil'):
    user = await usercollection.find_one({"email": email})
    if user:
        return user

    new_user = auth.UserCreate(
        email=email,
        first_name=first_name,
        last_name=last_name,
        password=''
    ).dict()
    new_user["verified"] = True
    new_user["disabled"] = False

    return new_user


secret_key = os.getenv("SESSION_SECRET_KEY", "default_fallback_secret_key")
origins = [
    "http://localhost:5173",
    "localhost:5173",
    "http://localhost:5173/signup",
    "http://localhost:5173/login",
    'https://backpackk.com/',
    'https://backpackk.com/signup',
    'https://backpackk.com/,login'
]
app.add_middleware(

    SessionMiddleware,
    secret_key=SECRET_KEY,

)
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from starlette.requests import Request

SECRET_KEY = os.getenv("SECRET_KEY", "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")
oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="https://accounts.google.com/o/oauth2/v2/auth",
    tokenUrl="https://oauth2.googleapis.com/token",
)


@app.get("/google")
async def google_auth(request: Request):
    google_auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"response_type=code&client_id=794713488480-8iqh9m6p3a93clvqrfrdjakt8q22movg.apps.googleusercontent.com&"
        f"redirect_uri=https://backpackkfast-fcvonqkgya-el.a.run.app/callback&scope=openid%20email%20profile"
    )
    return RedirectResponse(url=google_auth_url)


from httpx import AsyncClient


def generate_temp_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")


@app.get("/callback")
async def google_callback(code: str):
    # Retrieve the token from Google
    async with AsyncClient() as client:
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "code": code,
            "client_id": "794713488480-8iqh9m6p3a93clvqrfrdjakt8q22movg.apps.googleusercontent.com",
            "client_secret": "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",
            "redirect_uri": "https://backpackkfast-fcvonqkgya-el.a.run.app/callback",
            "grant_type": "authorization_code",
        }

        # Retrieve user info from Google
        token_response = await client.post(token_url, data=token_data)
        token_response_data = token_response.json()
        print(token_response_data)
        if "error" in token_response_data:
            raise HTTPException(status_code=400, detail="Failed to retrieve access token")

        access_token = token_response_data["access_token"]
        id_token = token_response_data["id_token"]

        # Get user info
        user_info_url = f"https://www.googleapis.com/oauth2/v1/userinfo?access_token={access_token}"
        user_info_response = await client.get(user_info_url)
        user_info = user_info_response.json()
        print(user_info)
        email = user_info['email']
        first_name = user_info['given_name']
        if 'family_name' in user_info:
            last_name = user_info['family_name']
            user = await find_or_create_user(email, first_name, last_name)
        else:
            user = await find_or_create_user(email, first_name)
        temp_token = generate_temp_token(email)
        response = RedirectResponse(f'https://backpackk.com/intermediate?token={temp_token}')
        return response





@app.post("/verifytemp")
async def verify_temp_token(param: VerifyToken):
    temptoken = param.temptoken
    payload = jwt.decode(temptoken, SECRET_KEY, algorithms=["HS256"])
    if payload:
        email = payload.get("email")
        existing_user = await usercollection.find_one({"email": email})
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        email = existing_user.get('email')
        name = existing_user.get('first_name')
        access_token = auth.create_access_token(
            data={"sub": email}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, 'first_name':name}




from jose import jwt

def generate_reset_token(email: str):
    payload = {
        "email": email,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def send_reset_email(email: str, token: str):
    sender_email = "nimishspslosal@gmail.com"
    sender_password = "axmdvfpmiewtzmsd"
    receiver_email = email

    subject = "Password Reset Request"
    reset_link = f"https://backpackkfast-fcvonqkgya-el.a.run.app/reset-password?token={token}"
    body = f"Please click on the following link to reset your password: {reset_link}"

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
        print("Password reset email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

@app.get("/forgot-password")
async def forgot_password(email: EmailStr, background_tasks: BackgroundTasks):
    user = await usercollection.find_one({"email": email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    token = generate_reset_token(user['email'])
    background_tasks.add_task(send_reset_email, user['email'], token)

    return {"message": "Password reset link sent to your email."}
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

@app.post("/forgot-password")
async def forgot_password(param: ForgotPasswordRequest, background_tasks: BackgroundTasks):

    user = await usercollection.find_one({"email": param.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    token = generate_reset_token(user['email'])
    background_tasks.add_task(send_reset_email, user['email'], token)

    return {"message": "Password reset link sent to your email."}
from resetpassmodel import resetpass

@app.get("/reset-password")
async def reset_password(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=404, detail="Invalid token.")
    response = RedirectResponse(f'https://backpackk.com/resetpassword?token={token}')
    return response
@app.post("/reset-password")
async def reset_password(param:resetpass):
    token = param.token
    new_password = param.new_password
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        email = payload.get("email")
        
        if not email:
            raise HTTPException(status_code=400, detail="Invalid token.")

        user = await usercollection.find_one({"email": email})
        if not user:
            raise HTTPException(status_code=404, detail="User not found.")

        hashed_password = auth.get_password_hash(new_password)
        await usercollection.update_one({"email": email}, {"$set": {"hashed_password": hashed_password}})
        
        return {"message": "Password has been reset successfully."}

    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Reset token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid reset token.")

import requests  
from typing import List
GOOGLE_API_KEY = "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
@app.get("/autocomplete" )
async def autocomplete_city_name(query: str = Query(..., min_length=1, description="City name to autocomplete")):

        url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
        params = {
            "input": query,
            "types": "(cities)",
            "key": GOOGLE_API_KEY,
        }
        print('hello')
        response = requests.get(url, params=params)
        predictions = response.json().get("predictions", [])

        city_names = [prediction["description"] for prediction in predictions]

        return city_names


@app.get("/getphoto")
async def get_photo(name: str,request:Request, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:

        print(name)
        photo_url = f"https://places.googleapis.com/v1/{name}/media?maxHeightPx=400&maxWidthPx=400&key=AIzaSyCzTbejaiLzlYUzDI8ZReYNgEF9UaS-X1E"
        if await request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        photo_response = requests.get(photo_url)
        print(photo_response.content)
        if photo_response.status_code != 200:
            raise HTTPException(status_code=photo_response.status_code, detail="Failed to retrieve photo.")

        image = Image.open(BytesIO(photo_response.content))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/jpeg")
from contactModel import contactModel
@app.post("/contactus")
async def contactus(param: contactModel):

    new_message = param.model_dump()
    print(new_message)
    collection= db['message']
    await collection.insert_one(new_message)
    return {'success': 'Message Received'}

@app.get("/reportbug")
async def contactus(message:str):


    print(message)
    collection= db['bugreport']
    await collection.insert_one({'message':message})
    return {'success': 'Thank You! Your contribution is valuable for us.'}