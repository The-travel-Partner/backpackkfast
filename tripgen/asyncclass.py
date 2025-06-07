import asyncio
import json
import random
import copy
import aiohttp.client
import requests
from io import BytesIO
from PIL import Image
from starlette.responses import Response
from unidecode import unidecode
import aiohttp
import base64
from bson.objectid import ObjectId
from datetime import datetime, timedelta

class retrieveplace:
    def __init__(self, placetypes, northeast, southwest, placesdb):
        self.placetypes = placetypes
        self.northeast = northeast
        self.southwest = southwest
        self.placesdb = placesdb
        self.semaphore = asyncio.Semaphore(15)  # Limit concurrent requests

    async def touristplace(self, session, place_type):
        async with self.semaphore:
            url = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk",
                "X-Goog-FieldMask": "places.id,places.location,places.rating,places.userRatingCount,places.types,places.photos,places.displayName,places.regularOpeningHours",
            }
            
            data = {
                "includedTypes": [place_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": self.northeast[0],
                            "longitude": self.northeast[1]
                        },
                        "radius": 50000.0
                    }
                }
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                data = await response.json()
                results = data.get('places', [])
                places = {}
                images_collection = self.placesdb['images']
                
                for place in results:
                    place_id = place.get('id')
                    name = place.get('displayName', {}).get('text', '')
                    lat = place.get('location', {}).get('latitude')
                    lng = place.get('location', {}).get('longitude')
                    rating = place.get('rating', 0)
                    user_ratings_total = place.get('userRatingCount', 0)
                    photo_refs = []
                    
                    if 'photos' in place:
                        photo_references = []
                        for photo in place['photos']:
                           
                            photo_ref = photo.get('name')
                     
                            if photo_ref:
                                photo_references.append(photo_ref)
                        
                        # Check if images for this place already exist in the database
                        existing_images = await images_collection.find({"place_id": place_id}).to_list(length=None)
                        
                        if existing_images:
                            # Use existing images
                            for image_doc in existing_images:
                                photo_refs.append(str(image_doc["_id"]))
                        else:
                            # Get actual photos using RetrievePhotos
                            if photo_references:
                                photos = RetrievePhotos(photo_references, place_id)
                                photo_responses = await photos.get_photos()
                       
                                for response in photo_responses:
                                    if response is not None:
                                        # Store image in separate collection
                                        image_doc = {
                                            "place_id": place_id,
                                            "image": base64.b64encode(response.body).decode('utf-8'),
                                            "content_type": "image/jpeg",
                                            "timestamp": datetime.now()
                                        }
                                        result = await images_collection.insert_one(image_doc)
                                        photo_refs.append(str(result.inserted_id))
                    
                    places[name] = {
                        "name": name,
                        "lat": lat,
                        "lng": lng,
                        "rating": rating,
                        "number": user_ratings_total,
                        "place_id": place_id,
                        "type": place_type,
                        "opening_hours": place.get('regularOpeningHours', {}).get('weekdayDescriptions', [])
                    }
                
                return places

    async def other(self, session, place_type):
        async with self.semaphore:
            url = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk",
                "X-Goog-FieldMask": "places.id,places.location,places.rating,places.userRatingCount,places.types,places.photos,places.displayName,places.regularOpeningHours",
            }
            
            data = {
                "includedTypes": [place_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": self.northeast[0],
                            "longitude": self.northeast[1]
                        },
                        "radius": 50000.0
                    }
                }
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                data = await response.json()
                results = data.get('places', [])
                places = {}
                images_collection = self.placesdb['images']
                
                for place in results:
                    place_id = place.get('id')
                    name = place.get('displayName', {}).get('text', '')
                    lat = place.get('location', {}).get('latitude')
                    lng = place.get('location', {}).get('longitude')
                    rating = place.get('rating', 0)
                    user_ratings_total = place.get('userRatingCount', 0)
                    photo_refs = []
                    
                    if 'photos' in place:
                        photo_references = []
                        for photo in place['photos']:
                            photo_ref = photo.get('name')
                            if photo_ref:
                                photo_references.append(photo_ref)
                        
                        # Check if images for this place already exist in the database
                        existing_images = await images_collection.find({"place_id": place_id}).to_list(length=None)
                        
                        if existing_images:
                            # Use existing images
                            for image_doc in existing_images:
                                photo_refs.append(str(image_doc["_id"]))
                        else:
                            # Get actual photos using RetrievePhotos
                            if photo_references:
                                photos = RetrievePhotos(photo_references, place_id)
                                photo_responses = await photos.get_photos()
                                
                                for response in photo_responses:
                                    if response is not None:
                                        # Store image in separate collection
                                        image_doc = {
                                            "place_id": place_id,
                                            "image": base64.b64encode(response.body).decode('utf-8'),
                                            "content_type": "image/jpeg",
                                            "timestamp": datetime.now()
                                        }
                                        result = await images_collection.insert_one(image_doc)
                                        photo_refs.append(str(result.inserted_id))
                    
                    places[name] = {
                        "name": name,
                        "lat": lat,
                        "lng": lng,
                        "rating": rating,
                        "number": user_ratings_total,
                        "place_id": place_id,
                        "type": place_type,
                        "opening_hours": place.get('regularOpeningHours', {}).get('weekdayDescriptions', [])
                    }
                
                return places

    async def food(self, session, place_type):
        async with self.semaphore:
            url = "https://places.googleapis.com/v1/places:searchNearby"
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk",
                "X-Goog-FieldMask": "places.id,places.location,places.rating,places.userRatingCount,places.types,places.photos,places.displayName,places.regularOpeningHours",
            }
            
            data = {
                "includedTypes": [place_type],
                "maxResultCount": 20,
                "locationRestriction": {
                    "circle": {
                        "center": {
                            "latitude": self.northeast[0],
                            "longitude": self.northeast[1]
                        },
                        "radius": 50000.0
                    }
                }
            }
            
            async with session.post(url, headers=headers, json=data) as response:
                data = await response.json()
                results = data.get('places', [])
                places = {}
                images_collection = self.placesdb['images']
                
                for place in results:
                    place_id = place.get('id')
                    name = place.get('displayName', {}).get('text', '')
                    lat = place.get('location', {}).get('latitude')
                    lng = place.get('location', {}).get('longitude')
                    rating = place.get('rating', 0)
                    user_ratings_total = place.get('userRatingCount', 0)
                    photo_refs = []
                    
                    if 'photos' in place:
                        photo_references = []
                        for photo in place['photos']:
                            photo_ref = photo.get('name')
                            if photo_ref:
                                photo_references.append(photo_ref)
                        
                        # Check if images for this place already exist in the database
                        existing_images = await images_collection.find({"place_id": place_id}).to_list(length=None)
                        
                        if existing_images:
                            # Use existing images
                            for image_doc in existing_images:
                                photo_refs.append(str(image_doc["_id"]))
                        else:
                            # Get actual photos using RetrievePhotos
                            if photo_references:
                                photos = RetrievePhotos(photo_references, place_id)
                                photo_responses = await photos.get_photos()
                                
                                for response in photo_responses:
                                    if response is not None:
                                        # Store image in separate collection
                                        image_doc = {
                                            "place_id": place_id,
                                            "image": base64.b64encode(response.body).decode('utf-8'),
                                            "content_type": "image/jpeg",
                                            "timestamp": datetime.now()
                                        }
                                        result = await images_collection.insert_one(image_doc)
                                        photo_refs.append(str(result.inserted_id))
                    
                    places[name] = {
                        "name": name,
                        "lat": lat,
                        "lng": lng,
                        "rating": rating,
                        "number": user_ratings_total,
                        "place_id": place_id,
                        "type": place_type,
                        "opening_hours": place.get('regularOpeningHours', {}).get('weekdayDescriptions', [])
                    }
                
                return places

    async def getAll(self):
        async with aiohttp.ClientSession() as session:
            tasks = []
            for place_type in self.placetypes:
                if place_type in ['tourist_attraction', 'museum', 'zoo']:
                    tasks.append(self.touristplace(session, place_type))
                elif place_type in ['night_club', 'bar', 'hindu_temple', 'church', 'mosque']:
                    tasks.append(self.other(session, place_type))
                elif place_type in ['restaurant', 'vegetarian_restaurant']:
                    tasks.append(self.food(session, place_type))
            
            results = await asyncio.gather(*tasks)
            
            final_result = {}
            for i, place_type in enumerate(self.placetypes):
                final_result[place_type] = results[i]
            
            return final_result
    def duplicateremover(self, js):
        res = {}
        dict_obj = json.loads(json.dumps(js))
        res = {k: v for k, v in dict_obj.items() if k not in res}
        return res

class place(retrieveplace):
    def __init__(self, placetypes, northeast, southwest, placesdb):
        super().__init__(placetypes, northeast, southwest, placesdb)
        self.tourist = {}
        self.museum = {}
        self.night_life = {}
        self.religiousplace = {}
        self.zoo = {}
        self.result = {}
        self.restaurant = {}
        self.veg_restaurant = {}
        self.types = placetypes
        self.session = aiohttp.ClientSession()

    async def touristattraction(self):
        index = self.types.index("tourist_attraction")
        res = await self.touristplace(self.session, self.types[index])
      
        self.tourist = super().duplicateremover(res)
        return self.tourist

    async def museums(self):
        index = self.types.index("museum")
        res = await super().other(self.session, self.types[index])
        self.museum = super().duplicateremover(res)
        return self.museum

    async def nightlife(self,place):
        res = await super().other(self.session,place_type=place)
        self.night_life.update(super().duplicateremover(res))
        return self.night_life

    async def religious(self, place):
        res = await super().other(self.session,place_type=place)
        self.religiousplace.update(super().duplicateremover(res))
        return self.religiousplace

    async def zoos(self):
        index = self.types.index("zoo")
        res = await super().other(self.session,self.types[index])
        self.zoo.update(super().duplicateremover(res))
        return self.zoo

    async def restro(self):
        index = self.types.index("restaurant")
        res = await super().other(self.session,self.types[index])
        self.restaurant = (super().duplicateremover(res))
        return self.restaurant

    async def veg_restro(self):
        index = self.types.index("vegetarian_restaurant")
        res = await super().other(self.session,self.types[index])
        self.veg_restaurant = (super().duplicateremover(res))
        return self.veg_restaurant

    async def getAll(self):
        for place in self.types:
            if place == "tourist_attraction":
               
                res = await self.touristattraction()
                self.tourist = copy.deepcopy(res)
                

                self.result['tourist_attraction'] = res
            elif place == "museum":
                res = await self.museums()
                self.museum = copy.deepcopy(res)
                
                self.result["museum"] = res
            elif place == "night_club" or place == "bar":
                res = await self.nightlife(place)
                self.night_life = copy.deepcopy(res)
               
                self.result[f'{place}'] = res
            elif place == "hindu_temple" or place == "mosque" or place == "church":

                res = await self.religious(place)
                self.religiousplace = copy.deepcopy(res)
               
                self.result[f"{place}"] = res
            elif place == "zoo":
                res = await self.zoos()
                self.zoo = copy.deepcopy(res)
               
                self.result[f"{place}"] = res
            elif place == "restaurant":
                res = await self.restro()
                self.restaurant = copy.deepcopy(res)
              
                self.result["restaurant"] = res
            elif place == "vegetarian_restaurant":
                res = await self.veg_restro()
                self.veg_restaurant = copy.deepcopy(res)
              
                self.result["vegetarian_restaurant"] = res
           
        return self.result


class RetrievePhotos:
    def __init__(self, photoreferences, place_id, max_concurrent_requests=5):
        self.photoreferences = photoreferences
        self.place_id = place_id
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def send_get_request(self, session, url):
        try:
            async with self.semaphore:
                async with session.get(url) as photo_response:
                    if photo_response.status == 200:
                        try:
                            if photo_response.headers.get('Content-Type') == 'application/json':
                                final_photo_url = (await photo_response.json())['photoUri']
                                async with session.get(final_photo_url) as final_response:
                                    if final_response.status == 200:
                                        return Response(
                                            content=await final_response.read(),
                                            media_type="image/jpeg"
                                        )
                            else:
                                return Response(
                                    content=await photo_response.read(),
                                    media_type="image/jpeg"
                                )
                        except Exception as e:
                            print(f"Error processing image data: {e}")
                            return None
                    else:
                        print(f"Error fetching photo: {photo_response}")
                        return None
        except Exception as e:
            print(f"Error in send_get_request: {e}")
            return None

    async def get_photos(self):
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.send_get_request(
                    session,
                    f"https://places.googleapis.com/v1/{ref}/media?maxHeightPx=400&maxWidthPx=400&key=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
                )
                for ref in self.photoreferences
            ]
            results = await asyncio.gather(*tasks)
            return [r for r in results if r is not None]