import base64
import json
import time
from urllib.request import Request
import asyncio

import googlemaps
import pandas as pd
import redis
import requests
import vertexai

from jose import jwt
from fastapi import FastAPI, Depends, HTTPException, status, Query, Request
from fastapi import FastAPI, Response
from fastapi.security import OAuth2PasswordRequestForm, OAuth2AuthorizationCodeBearer
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse, HTMLResponse

from tripgen.bestplacesModel import bestPlacesModel
from tripgen.getplacesModel import getplacesModel
from tripgen.placesDBClass import placesDBClass
from tripgen.placesRetrievenew import placesRetrieve
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
from Redis import RedisManager
from tripgen.asyncclass import place

mongostr = "mongodb+srv://admin:C5Qt4vWNogmRSlVi@backpackk.tmkdask.mongodb.net/"
client = AsyncIOMotorClient(mongostr)
db = client['backpackk']
usercollection = db['users']

from authenticate.authentication import authenticate

SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 43800
apikey= "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
auth = authenticate(secretkey=SECRET_KEY, algorithm=ALGORITHM, usercollection=usercollection)
origin_url= "https://backpackk-cloud.el.r.appspot.com/"



# Initialize Redis manager
redisClient = RedisManager()

app = FastAPI()
origins = [

    'https://backpackk.com',
    'https://backpackk.com/signup',
    'https://backpackk.com/login',
    "backpackk.com",
    "http://localhost:4173",
    "localhost:4173",
    "http://localhost:4173/signup",
    "http://localhost:4173/login",
    "backpackksveltekit-76dfccaa6eb3.herokuapp.com",
    "https://backpackk-cloud.el.r.appspot.com",
    "backpackk-cloud.el.r.appspot.com",
    "http://localhost:5173",
    "localhost:5173",
    "*"
]
app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
)


async def current_active_user_dependency(current_user: auth.UserInDB = Depends(auth.get_current_user)):
    return await auth.get_current_active_user(current_user)


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


app.add_middleware(

    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.post("/token", response_model=auth.Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    print(form_data.username)
    print(form_data.password)
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
    response = JSONResponse({"access_token": access_token, "token_type": "bearer", "first_name": name})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


def generate_verification_token(length=6):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


app.add_middleware(

    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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


app.add_middleware(

    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


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
    response = JSONResponse({"message": "User registered successfully. Please check your email to verify your account."})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


@app.get("/verify/{user_id}/{token}")
async def verify_user(user_id: str, token: str):
    user = await usercollection.find_one({"_id": ObjectId(user_id), "verification_token": token})
    if user:
        await usercollection.update_one({"_id": ObjectId(user_id)},
                                        {"$set": {"verified": True}, "$unset": {"verification_token": ""}})
        response = JSONResponse({"message": "Email verified successfully!"})
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response

    else:
        raise HTTPException(status_code=400, detail="Verification failed: Invalid token or user ID.",
                            headers={'Access-Control-Allow-Origin': "https://backpackkfast-fcvonqkgya-el.a.run.app"})


@app.get("/users/me", response_model=auth.User)
async def read_users_me(current_user: auth.UserInDB = Depends(current_active_user_dependency)):

    response = JSONResponse(current_user.model_dump())
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


@app.post('/tripgenerator')
async def generator(param: tripgenModel, request: Request,
                    current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        city_name = param.city_name
        place_types = param.place_types
        no_of_days = param.no_of_days
        weekdays = param.weekdays
        travel_schedule = param.travel_schedule
        print(place_types)
        placesCollection = db['placesdata']
        city = await placesCollection.find_one({"city_name": city_name})

        # Try to get cached data from Redis, with fallback to database
        js = redisClient.get(current_user.email)
        if js is None:
            print("⚠️ Redis unavailable or no cached data found - fetching from database")
            # Fallback: Try to reconstruct data from database
            if city:
                # Create a basic structure from database data
                final_js = {
                    'email': current_user.email,
                    'places': city['places']
                }
                print("✅ Using database fallback for places data")
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="No cached trip data available and city not found in database. Please generate places first using /getplaces endpoint."
                )
        else:
            try:
                final_js = json.loads(js)
                print("✅ Using cached data from Redis")
            except json.JSONDecodeError:
                print("❌ Invalid JSON in Redis cache, falling back to database")
                if city:
                    final_js = {
                        'email': current_user.email,
                        'places': city['places']
                    }
                else:
                    raise HTTPException(
                        status_code=400, 
                        detail="Corrupted cache data and city not found in database. Please generate places first."
                    )
        
        df_sorted_json = final_js['places']
        df_sorted = pd.DataFrame(df_sorted_json)
        trip = TripCreator(request=request, city_name=city_name, place_types=place_types, no_of_days=no_of_days, useremail=current_user.email, placesdb=db, df_sorted=df_sorted, weekday=weekdays, travel_schedule=travel_schedule)
        new_trip = await trip.create_trip()

        usertripCollection = db['user_trips']

        finaltrip = {'city_name': city_name, 'place_types': place_types, 'no_of_days': no_of_days,"travel_schedule":travel_schedule, 'trip': new_trip}
        user = await usertripCollection.find_one({"email": current_user.email})

        if user:
            result = await usertripCollection.update_one(
                {'email': current_user.email},
                {'$push': {'trips': finaltrip}}
            )
            print(result)
        else:
            await usertripCollection.insert_one(
                {'email': current_user.email, 'trips': [finaltrip]})

        response = JSONResponse(finaltrip)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response


import os


async def find_or_create_user(email, first_name, last_name='nil'):
    user = await usercollection.find_one({"email": email})
    hashed_password = auth.get_password_hash("default")
    if user:
        return user
    else:
        new_user = auth.UserCreate(
            email=email,
            first_name=first_name,
            last_name=last_name,
            password=''
        ).dict()
        new_user["hashed_password"] = hashed_password
        new_user["verified"] = True
        new_user["disabled"] = False
        user = await usercollection.insert_one(new_user)
        return new_user


secret_key = os.getenv("SESSION_SECRET_KEY", "default_fallback_secret_key")

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
    return RedirectResponse(url=google_auth_url,
                            headers={'Access-Control-Allow-Origin': "https://backpackkfast-fcvonqkgya-el.a.run.app"})


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
        response.headers["Access-Control-Allow-Origin"] = origin_url
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
        response = JSONResponse({"access_token": access_token, 'first_name': name})
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response


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

    response = JSONResponse({"message": "Password reset link sent to your email."})
    response.headers["Access-Control-Allow-Origin"] =origin_url
    return response


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@app.post("/forgot-password")
async def forgot_password(param: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    user = await usercollection.find_one({"email": param.email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    token = generate_reset_token(user['email'])
    background_tasks.add_task(send_reset_email, user['email'], token)

    response = JSONResponse({"message": "Password reset link sent to your email."})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


from resetpassmodel import resetpass


@app.get("/reset-password")
async def reset_password(token: str):
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    email = payload.get("email")
    if not email:
        raise HTTPException(status_code=404, detail="Invalid token.")
    response = RedirectResponse(f'https://backpackk.com/resetpassword?token={token}')
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


@app.post("/reset-password")
async def reset_password(param: resetpass):
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

        response = JSONResponse({"message": "Password has been reset successfully."})
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response

    except jwt.ExpiredSignatureError:
        response = HTTPException(status_code=400, detail="Reset token has expired.")
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response
    except jwt.InvalidTokenError:
        response = HTTPException(status_code=400, detail="Invalid reset token.")
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response


import requests
from typing import List

GOOGLE_API_KEY = "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"


@app.get("/autocomplete")
async def autocomplete_city_name(query: str = Query(..., min_length=1, description="City name to autocomplete")):
    url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    params = {
        "input": query,
        "types": "(cities)",
        "components": f"country:IN",  # Restrict results to a specific country
        "key": GOOGLE_API_KEY,
    }
    print('hello')
    response = requests.get(url, params=params)
    predictions = response.json().get("predictions", [])

    city_names = [prediction["description"] for prediction in predictions]

    response = JSONResponse({'cities': city_names})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response

@app.post("/number/of/available/days")
async def numberofdays(param: tripgenModel, request: Request,
                    current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    city_name = param.city_name
    place_types = param.place_types
    travel_schedule = param.travel_schedule
    weekdays = param.weekdays
    final_js = {}
    # Try to get cached data from Redis, with fallback to database
    js = redisClient.get(current_user.email)
    if js is None:
        print("⚠️ Redis unavailable or no cached data found - fetching from database")
        # Fallback: Try to get data from database
        placesCollection = db['placesdata']
        city = await placesCollection.find_one({"city_name": city_name})
        if city:
            final_js = {
                'email': current_user.email,
                'places': city['places']
            }
            print("✅ Using database fallback for places data")
        else:
            raise HTTPException(
                status_code=400, 
                detail="No cached trip data available and city not found in database. Please generate places first using /getplaces endpoint."
            )
    else:
        try:
            final_js = json.loads(js)
            print("✅ Using cached data from Redis")
        except json.JSONDecodeError:
            print("❌ Invalid JSON in Redis cache, falling back to database")
            placesCollection = db['placesdata']
            city = await placesCollection.find_one({"city_name": city_name})
            if city:
                final_js = {
                    'email': current_user.email,
                    'places': city['places']
                }
            else:
                raise HTTPException(
                    status_code=400, 
                    detail="Corrupted cache data and city not found in database. Please generate places first."
                )
    
    df_sorted_json = final_js['places']
    
    df_sorted = pd.DataFrame(df_sorted_json)

    df_sorted.to_csv('test.csv')
    days = TripCreator(request=request, city_name=city_name, place_types=place_types, travel_schedule=travel_schedule, weekday=weekdays, df_sorted=df_sorted, useremail=current_user.email, placesdb=db, no_of_days=120)
    number = await days.create_trip()
    print(number)
    response = JSONResponse({"Number": number})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response

@app.post("/getplaces")
async def getplaces(param: getplacesModel, request: Request,current_user: auth.UserInDB = Depends(current_active_user_dependency)):
   if current_user:
       city_name = param.city_name
       place_types = param.place_types
       useremail = current_user.email

       placesCollection = db['placesdata']
       city = await placesCollection.find_one({"city_name": city_name})
       places = placesRetrieve(request=request, city_name=city_name, place_types=place_types,useremail=useremail,placesdb=db)
       finalPlaces,placesdata =await  places.getplaces()
       print(json.dumps(finalPlaces.to_json(), indent=4)) 
       timestamp = f"{str(time.localtime().tm_mon)},{str(time.localtime().tm_year)}"
       places = {'city_name': city_name, "timestamp": timestamp, 'places': placesdata}
       if city is None:
           if placesdata is not None:
               await placesCollection.insert_one(places)

       bestPlacesModel.modelData = json.loads(finalPlaces.to_json())
       final = {"email": current_user.email, "places": json.loads(finalPlaces.to_json())}
       
       # Try to cache data in Redis, continue even if Redis is unavailable
       cache_success = redisClient.setex(current_user.email,3600, json.dumps(final))
       if cache_success:
           print("✅ Successfully cached places data in Redis")
       else:
           print("⚠️ Failed to cache places data in Redis - continuing without cache")
       

       response = JSONResponse({"Places":"True"})
       response.headers["Access-Control-Allow-Origin"] = origin_url
       return response

@app.get("/getplaces/all")
async def getplaces(cityname: str, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        city_name = cityname
        placesCollection = db['placesdata']
        city = await placesCollection.find_one({"city_name": city_name})

        if city is None:
            gmaps = googlemaps.Client(key='AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk')
            geocode_result = gmaps.geocode(city_name)
            print(geocode_result[0]['geometry']['location'].keys())

            bounds = geocode_result[0]['geometry']['bounds']
            latt = geocode_result[0]['geometry']['location']['lat']
            lngg = geocode_result[0]['geometry']['location']['lng']

            northeast = bounds['northeast']
            southwest = bounds['southwest']
            print(northeast)

            northeast = (northeast['lat'], northeast['lng'])
            southwest = (southwest['lat'], southwest['lng'])
            print('Northeast coordinates:', northeast)
            print('Southwest coordinates:', southwest)
            places = place(placetypes=[
                        "museum",
                        "tourist_attraction",
                        "hindu_temple",
                        "zoo",
                        "night_club",
                        "bar",
                        "restaurant",
                        "vegetarian_restaurant"
                    ], northeast=northeast,southwest=southwest,placesdb=db)
            result = await places.getAll()
            timestamp = f"{str(time.localtime().tm_mon)},{str(time.localtime().tm_year)}"
            places = {'city_name': city_name, "timestamp": timestamp, 'places': result}
            if city is None:
                if result is not None:
                    await placesCollection.insert_one(places)
        else:
            result = json.dumps(city['places'])
        final = {"email": current_user.email, "places": json.loads(result)}
 

        # Try to cache data in Redis, continue even if Redis is unavailable
        cache_success = redisClient.setex(current_user.email, 3600, json.dumps(final))
        if cache_success:
            print("✅ Successfully cached places data in Redis")
        else:
            print("⚠️ Failed to cache places data in Redis - continuing without cache")

        print(type(city))
        return JSONResponse(city['places'])
@app.get("/getphoto")
async def get_photo(name: str, request: Request, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:

        print(name)
        photo_url = f"https://places.googleapis.com/v1/{name}/media?maxHeightPx=400&maxWidthPx=400&key=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
        if await request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        photo_response = requests.get(photo_url)
        print(photo_response.text)
        if photo_response.headers.get('Content-Type') == 'application/json':
            final_photo_url = photo_response.json()['photoUri']
            final = requests.get(final_photo_url)
            image = Image.open(BytesIO(final.content))
        else:
            image = Image.open(BytesIO(photo_response.content))
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        response = Response(content=img_byte_arr, media_type="image/jpeg")
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response


from contactModel import contactModel
from tripgen.asyncclass import RetrievePhotos
from tripgen.tripgenModel import getPhotos
@app.post("/getphotos")
async def getPhotos(param: getPhotos):
    photoref = param.photoref
    print(photoref)
    photos = RetrievePhotos(photoref)
    images = await photos.get_photos()
    image_bytes=''
    base64_images = []
    for response in images:

        if hasattr(response, 'body'):
            image_bytes = response.body

        elif hasattr(response, 'content'):
            image_bytes = response.content


        # Create complete data URI
        print(type(image_bytes))

        base64_images.append({
            "image": image_bytes,
            "content_type": "image/jpeg"  # Adjust if using different image format
        })
    print(len(base64_images))
    return {"images": base64_images}
from tripgen.tripgenModel import getPhotosByPlaceId
@app.post("/getphotos/byplaceid")
async def getPhotosByPlaceId(param: getPhotosByPlaceId):
    place_id = param.place_id
    print(f"Retrieving images for place_id: {place_id}")
    
    # Access the images collection
    images_collection = client.backpackk.images
    
    # Find all images for the given place_id
    cursor = images_collection.find({"place_id": place_id})
    images = await cursor.to_list(length=None)
    
    # Format the response
    base64_images = []
    for image_doc in images:
        base64_images.append({
            "image": image_doc["image"],
            "content_type": image_doc["content_type"]
        })
    
    print(f"Found {len(base64_images)} images for place_id: {place_id}")
    response = JSONResponse({"images": base64_images})
    return response

@app.post("/contactus")
async def contactus(param: contactModel):
    new_message = param.model_dump()
    print(new_message)
    collection = db['message']
    await collection.insert_one(new_message)
    response = JSONResponse({'success': 'Message Received'})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response


@app.get("/reportbug")
async def contactus(message: str):
    print(message)
    collection = db['bugreport']
    await collection.insert_one({'message': message})
    response = JSONResponse({'success': 'Thank You! Your contribution is valuable for us.'})
    response.headers["Access-Control-Allow-Origin"] = origin_url
    return response

@app.get("/generate-map-blob")
async def generate_map_blob(placename:str, location: str = Query(..., min_length=1),current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        src = f"https://maps.googleapis.com/maps/api/staticmap?center={location}&zoom=21&size=600x400&maptype=roadmap&markers=color:red%7Clabel:{placename}%7C{location}&key={apikey}"
        blob_data = BytesIO(src.encode('utf-8'))
        response= StreamingResponse(blob_data, media_type="text/plain")
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response
    else:
        response= JSONResponse({"error": "Unauthorised"}, status_code=403)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response

@app.get('/generate-map-blob/streetview')
async def streetview(location:str, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        src = f"https://www.google.com/maps/embed/v1/streetview?location={location}&key=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
        response= Response(src)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response
    else:
        response= JSONResponse({"error": "Unauthorised"}, status_code=403)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response
@app.get('/generate-map-blob/view')
async def view(placeid: str, current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        src = f"https://www.google.com/maps/embed/v1/place?q=place_id:{placeid}&key=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
        response= Response(src)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response
    else:
        response= JSONResponse({"error": "Unauthorised"}, status_code=403)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response

from vertexai.generative_models import GenerativeModel, Part, Tool
import google.genai as genai
import vertexai.preview.generative_models as generative_models
import vertexai
@app.get("/placedescription")
async def get_place_description(placename:str, cityname:str= Query(..., min_length=1),current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    print(current_user)
    if current_user:
        # Create a unique key for Redis
        print('cityname',cityname)
        print('placename',placename)
        redis_key = f"description:{cityname}:{placename}"
        
        # First check Redis cache
        cached_description = redisClient.get(redis_key)
        if cached_description:
            print('✅ Using cached description from Redis:', cached_description)
            # Return cached description from Redis
            response = JSONResponse(content={"description": cached_description})
            response.headers["Access-Control-Allow-Origin"] = origin_url
            return response

        # If not in Redis, check MongoDB
        descriptions_collection = db['descriptions']
        existing_description = await descriptions_collection.find_one({
            "place_name": placename,
            "city_name": cityname
        })
        
        if existing_description:
            # Try to cache the MongoDB description in Redis (expire after 7 days)
            cache_success = redisClient.setex(redis_key, 7 * 24 * 60 * 60, existing_description["description"])
            if cache_success:
                print("✅ Successfully cached MongoDB description in Redis")
            else:
                print("⚠️ Failed to cache MongoDB description in Redis - continuing without cache")
            
            # Return existing description from MongoDB
            response = JSONResponse(content={"description": existing_description["description"]})
            response.headers["Access-Control-Allow-Origin"] = origin_url
            return response

        # If no existing description, generate new one using Gemini
        from google.genai import types
        client = genai.Client(
            vertexai=True,
            project="backpackk-cloud",
            location="global",
        )
        
        tools = [
            types.Tool(google_search=types.GoogleSearch()),
        ]

        system_instruction = """Provide a detailed, informative description of the place in 3-4 sentences. Include:
1. Basic information (location, when it was built/established if historical)
2. Key architectural or design features
3. Historical or cultural significance
4. Any unique characteristics or interesting facts
Keep the tone informative and engaging.

Keep it extremely short, 30 words max"""

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=f'Provide a detailed description of {placename} in {cityname}. Focus on its history, architecture, significance, and unique features.')
                ]
            ),
        ]

        generate_content_config = types.GenerateContentConfig(
            temperature=0.7,
            top_p=0.95,
            max_output_tokens=1024,
            response_modalities=["TEXT"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )
            ],
            tools=tools,
            system_instruction=[types.Part.from_text(text=system_instruction)],
        )

        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=contents,
                config=generate_content_config
            )

            if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
                return JSONResponse(content={"description": "Could not generate description"}, status_code=500)

            description = response.candidates[0].content.parts[0].text
            print(description)

            # Save the generated description to MongoDB
            description_doc = {
                "place_name": placename,
                "city_name": cityname,
                "description": description,
                "timestamp": datetime.now()
            }
            await descriptions_collection.insert_one(description_doc)

            # Try to cache the new description in Redis (expire after 7 days)
            cache_success = redisClient.setex(redis_key, 7 * 24 * 60 * 60, description)
            if cache_success:
                print("✅ Successfully cached generated description in Redis")
            else:
                print("⚠️ Failed to cache generated description in Redis - continuing without cache")

            # Return the generated description
            response = JSONResponse(content={"description": description})
            response.headers["Access-Control-Allow-Origin"] = origin_url
            return response
        except Exception as e:
            print(f"Error generating description: {str(e)}")
            return JSONResponse(content={"description": f"Error generating description: {str(e)}"}, status_code=500)
    else:
        response = JSONResponse({"error": "Unauthorised"}, status_code=403)
        response.headers["Access-Control-Allow-Origin"] = origin_url
        return response


# Sample map data
map_data = {
    "sets": [
        {
            "color": "#FF5733",  # Red route
            "points": [
                {"lat": 37.7749, "lng": -122.4194},  # San Francisco
                {"lat": 34.0522, "lng": -118.2437},  # Los Angeles
            ],
        },
        {
            "color": "#33C3FF",  # Blue route
            "points": [
                {"lat": 40.7128, "lng": -74.0060},  # New York
                {"lat": 42.3601, "lng": -71.0589},  # Boston
            ],
        },
    ]
}

@app.get('/lasttrip')
async def get_feed(current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    if current_user:
        collection = db['user_trips']
        print(current_user.email)
        user = await collection.find_one({'email':current_user.email})
        if not user:
            return {"error": "User not found or no trips created."}
        trips = user.get("trips")
        print(trips[-1])
        if trips:
            # Retrieve the last trip added
            last_trip = trips[-1]



            return JSONResponse(content=last_trip, status_code=200)
        else:
            return {"error": "No trips available for this user."}



import aiohttp

class NearbyPlacesRequest(BaseModel):
    latitude: float
    longitude: float
    placeid: str
    radius: int = 3000  # Default radius in meters
    max_results: int = 20  # Default maximum number of results
    include_photos: bool = True  # Whether to include photos in the response

class NearbyLocation:
    def __init__(self, place_id: str, name: str, place_type: str, location: dict, rating: float = None, 
                 vicinity: str = None, photos: list = None):
        self.place_id = place_id
        self.name = name
        self.place_type = place_type
        self.location = location
        self.rating = rating
        self.vicinity = vicinity
        # Ensure only one photo is stored
        self.photos = photos[:1] if photos else []
    
    def to_dict(self):
        return {
            "place_id": self.place_id,
            "name": self.name,
            "place_type": self.place_type,
            "location": self.location,
            "rating": self.rating,
            "vicinity": self.vicinity,
            "photos": self.photos[:1] if self.photos else []  # Ensure only one photo is returned
        }

from tripgen.asyncclass import RetrievePhotos

class PlaceDetailsRequest(BaseModel):
    place_id: str
    include_photos: bool = True

@app.post("/place")
async def get_place_details(params: PlaceDetailsRequest,
                          current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    try:
        # Access MongoDB collections
        places_collection = db['places']  # Changed to places folder
        images_collection = db['images']  # Changed to images folder
        
        # Check if place exists in MongoDB
        existing_place = await places_collection.find_one({"place_id": params.place_id})
        
        if existing_place:
            # Remove _id field
            existing_place.pop('_id', None)
            
            # Get the associated image from images collection
            if params.include_photos:
                existing_place["photos"] = [] # Initialize as list
                image_cursor = images_collection.find({"place_id": params.place_id})
                async for image_doc in image_cursor:
                    image_doc.pop('_id', None)
                    existing_place["photos"].append({
                        "image": image_doc["image"],
                        "content_type": image_doc["content_type"]
                    })
            else:
                existing_place["photos"] = []
            
            # Return existing place data
            print(existing_place)
            if existing_place.get("timestamp"):
                existing_place["timestamp"] = existing_place["timestamp"].isoformat()
            return JSONResponse(content=existing_place)
        
        # If place doesn't exist, fetch it
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": apikey,
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.rating,places.userRatingCount,places.photos,places.types,places.formattedAddress"
        }
        
        # Fetch the specific place details
        place_url = f"https://places.googleapis.com/v1/places/{params.place_id}"
        async with aiohttp.ClientSession() as session:
            async with session.get(place_url, headers=headers) as response:
                place_response = await response.json()
                
                if "error" in place_response:
                    raise HTTPException(status_code=404, detail="Place not found")
                
                # Process and store the place
                place = place_response
                name = place.get("displayName", {}).get("text", "")
                location = place.get("location", {})
                rating = place.get("rating")
                vicinity = place.get("formattedAddress")
                
                # Get photo references if they exist
                photo_refs = []
                if "photos" in place:
                    for photo in place["photos"]:
                        photo_ref = photo.get("name")
                        if photo_ref:
                            photo_refs.append(photo_ref)
                
                # Initialize location photos list
                location_photos = []
                
                # If photos are requested, fetch them
                if params.include_photos and photo_refs:
                    photos_retriever = RetrievePhotos(photo_refs, params.place_id) # Pass all refs
                    photo_responses = await photos_retriever.get_photos()
                    
                    for photo_resp_item in photo_responses: # Iterate through all responses
                        if photo_resp_item is not None and hasattr(photo_resp_item, 'body'):
                            # Store image in images collection
                            image_doc_content = base64.b64encode(photo_resp_item.body).decode('utf-8')
                            image_doc = {
                                "place_id": params.place_id,
                                "image": image_doc_content,
                                "content_type": "image/jpeg", # Assuming JPEG, adjust if necessary
                                "timestamp": datetime.now().isoformat() # Store as ISO string
                            }
                            # Check if this specific image (by content or a unique photo_ref part) already exists to avoid duplicates
                            # This part might need more sophisticated handling if Google's photo_refs are not unique enough for direct use
                            # For now, we'll insert, but duplicate image data might occur if not handled.
                            await images_collection.insert_one(image_doc)
                            
                            location_photos.append({
                                "image": image_doc_content,
                                "content_type": image_doc["content_type"]
                            })
                
                # Store place in places collection without photos
                place_doc = {
                    "place_id": params.place_id,
                    "name": name,
                    "place_type": "attraction",  # Default type, will be updated if different
                    "location": location,
                    "rating": rating,
                    "vicinity": vicinity,
                    "timestamp": datetime.now()
                }
                await places_collection.insert_one(place_doc)
                
                # Add photos to the response
                place_doc["photos"] = location_photos
                if place_doc.get("timestamp"):
                    place_doc["timestamp"] = place_doc["timestamp"].isoformat()
                
                return JSONResponse(content=place_doc)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching place details: {str(e)}")

@app.post("/nearby-places")
async def get_nearby_places(params: NearbyPlacesRequest, 
                           current_user: auth.UserInDB = Depends(current_active_user_dependency)):
    try:
        # Initialize results
        print(params.json())
        attractions = []
        restaurants = []
        
        # Access MongoDB collections
        places_collection = client.backpackk.places
        images_collection = client.backpackk.images
        
     
        
        # Now fetch nearby places using the provided coordinates
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": apikey,
            "X-Goog-FieldMask": "places.id,places.displayName,places.location,places.rating,places.userRatingCount,places.photos,places.types,places.formattedAddress"
        }
        
        places_url = "https://places.googleapis.com/v1/places:searchNearby"
        
        async with aiohttp.ClientSession() as session:
            # Fetch attractions
            attraction_data = {
                "includedTypes": ["tourist_attraction"],
                "maxResultCount": params.max_results // 2,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": params.latitude,
                            "longitude": params.longitude
                        },
                        "radius": params.radius
                    }
                }
            }
            
            async with session.post(places_url, headers=headers, json=attraction_data) as response:
                attraction_response = await response.json()
                
                if "places" in attraction_response:
                    for place in attraction_response["places"]:
                        place_id = place.get("id")
                        
                        # Check if place exists in MongoDB
                        existing_place = await places_collection.find_one({"place_id": place_id})
                        
                        if existing_place:
                            location_obj = NearbyLocation(
                                place_id=existing_place["place_id"],
                                name=existing_place["name"],
                                place_type="attraction",
                                location=existing_place["location"],
                                rating=existing_place.get("rating"),
                                vicinity=existing_place.get("vicinity"),
                                photos=existing_place.get("photos", [])[:1]
                            )
                        else:
                            name = place.get("displayName", {}).get("text", "")
                            location = place.get("location", {})
                            rating = place.get("rating")
                            vicinity = place.get("formattedAddress")
                            
                            photo_refs = []
                            if "photos" in place:
                                for photo in place["photos"]:
                                    photo_ref = photo.get("name")
                                    if photo_ref:
                                        photo_refs.append(photo_ref)
                            
                            location_photos = []
                            if params.include_photos and photo_refs:
                                photos_retriever = RetrievePhotos(photo_refs, place_id) # Pass all refs
                                photo_responses = await photos_retriever.get_photos()
                                
                                for photo_resp_item in photo_responses: # Iterate through all responses
                                    if photo_resp_item is not None and hasattr(photo_resp_item, 'body'):
                                        # Store image in images collection
                                        image_doc_content = base64.b64encode(photo_resp_item.body).decode('utf-8')
                                        image_doc = {
                                            "place_id": place_id,
                                            "image": image_doc_content,
                                            "content_type": "image/jpeg", # Assuming JPEG, adjust if necessary
                                            "timestamp": datetime.now().isoformat() # Store as ISO string
                                        }
                                        # Check if this specific image (by content or a unique photo_ref part) already exists to avoid duplicates
                                        # This part might need more sophisticated handling if Google's photo_refs are not unique enough for direct use
                                        # For now, we'll insert, but duplicate image data might occur if not handled.
                                        await images_collection.insert_one(image_doc)
                                        
                                        location_photos.append({
                                            "image": image_doc_content,
                                            "content_type": image_doc["content_type"]
                                        })
                            
                            location_obj = NearbyLocation(
                                place_id=place_id,
                                name=name,
                                place_type="attraction",
                                location=location,
                                rating=rating,
                                vicinity=vicinity,
                                photos=location_photos
                            )
                            
                            place_doc = {
                                "place_id": place_id,
                                "name": name,
                                "place_type": "attraction",
                                "location": location,
                                "rating": rating,
                                "vicinity": vicinity,
                                "photos": location_photos,
                                "timestamp": datetime.now()
                            }
                            await places_collection.insert_one(place_doc)
                        
                        attractions.append(location_obj.to_dict())
            
            # Fetch restaurants (similar logic as attractions)
            restaurant_data = {
                "includedTypes": ["restaurant"],
                "maxResultCount": params.max_results // 2,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": params.latitude,
                            "longitude": params.longitude
                        },
                        "radius": params.radius
                    }
                }
            }
            
            async with session.post(places_url, headers=headers, json=restaurant_data) as response:
                restaurant_response = await response.json()
                
                if "places" in restaurant_response:
                    for place in restaurant_response["places"]:
                        place_id = place.get("id")
                        
                        # Check if place exists in MongoDB
                        existing_place = await places_collection.find_one({"place_id": place_id})
                        
                        if existing_place:
                            location_obj = NearbyLocation(
                                place_id=existing_place["place_id"],
                                name=existing_place["name"],
                                place_type="restaurant",
                                location=existing_place["location"],
                                rating=existing_place.get("rating"),
                                vicinity=existing_place.get("vicinity"),
                                photos=existing_place.get("photos", [])[:1]
                            )
                        else:
                            name = place.get("displayName", {}).get("text", "")
                            location = place.get("location", {})
                            rating = place.get("rating")
                            vicinity = place.get("formattedAddress")
                            
                            photo_refs = []
                            if "photos" in place:
                                for photo in place["photos"]:
                                    photo_ref = photo.get("name")
                                    if photo_ref:
                                        photo_refs.append(photo_ref)
                            
                            location_photos = []
                            if params.include_photos and photo_refs:
                                photos_retriever = RetrievePhotos(photo_refs, place_id) # Pass all refs
                                photo_responses = await photos_retriever.get_photos()
                                
                                for photo_resp_item in photo_responses: # Iterate through all responses
                                    if photo_resp_item is not None and hasattr(photo_resp_item, 'body'):
                                        # Store image in images collection
                                        image_doc_content = base64.b64encode(photo_resp_item.body).decode('utf-8')
                                        image_doc = {
                                            "place_id": place_id,
                                            "image": image_doc_content,
                                            "content_type": "image/jpeg", # Assuming JPEG, adjust if necessary
                                            "timestamp": datetime.now().isoformat() # Store as ISO string
                                        }
                                        # Check if this specific image (by content or a unique photo_ref part) already exists to avoid duplicates
                                        # This part might need more sophisticated handling if Google's photo_refs are not unique enough for direct use
                                        # For now, we'll insert, but duplicate image data might occur if not handled.
                                        await images_collection.insert_one(image_doc)
                                        
                                        location_photos.append({
                                            "image": image_doc_content,
                                            "content_type": image_doc["content_type"]
                                        })
                            
                            location_obj = NearbyLocation(
                                place_id=place_id,
                                name=name,
                                place_type="restaurant",
                                location=location,
                                rating=rating,
                                vicinity=vicinity,
                                photos=location_photos
                            )
                            
                            place_doc = {
                                "place_id": place_id,
                                "name": name,
                                "place_type": "restaurant",
                                "location": location,
                                "rating": rating,
                                "vicinity": vicinity,
                                "photos": location_photos,
                                "timestamp": datetime.now()
                            }
                            await places_collection.insert_one(place_doc)
                        
                        restaurants.append(location_obj.to_dict())
        
        return {
            "attractions": attractions,
            "restaurants": restaurants,
            "total_results": len(attractions) + len(restaurants)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching nearby places: {str(e)}")
        




    
    
    
