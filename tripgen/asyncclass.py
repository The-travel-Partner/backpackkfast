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

async def process_photos_background(photo_references, place_id, images_collection):
    try:
        # Process photos in smaller chunks
        chunk_size = 5
        for i in range(0, len(photo_references), chunk_size):
            chunk = photo_references[i:i + chunk_size]
            photos = RetrievePhotos(chunk, place_id)
            photo_responses = await photos.get_photos()
            
            for response in photo_responses:
                if response is not None:
                    image_doc = {
                        "place_id": place_id,
                        "image": base64.b64encode(response.body).decode('utf-8'),
                        "content_type": "image/jpeg",
                        "timestamp": datetime.now()
                    }
                    await images_collection.insert_one(image_doc)
            
            # Add delay between chunks
            if i + chunk_size < len(photo_references):
                await asyncio.sleep(2)  # 2 second delay between chunks
                
    except Exception as e:
        print(f"Error in background photo processing: {e}")

class retrieveplace:
    def __init__(self, placetypes, northeast, southwest, placesdb):
        self.placetypes = placetypes
        self.northeast = northeast
        self.southwest = southwest
        self.placesdb = placesdb
        self.semaphore = asyncio.Semaphore(15)
        self.semaphore2 = asyncio.Semaphore(5)
        self.background_tasks = set()
        self.photo_processing_status = {}
        self.connection_semaphore = asyncio.Semaphore(50)  # Limit concurrent connections

    async def process_single_place(self, place, place_type, images_collection):
        try:
            place_id = place.get('id')
            name = place.get('displayName', {}).get('text', '')
            lat = place.get('location', {}).get('latitude')
            lng = place.get('location', {}).get('longitude')
            rating = place.get('rating', 0)
            user_ratings_total = place.get('userRatingCount', 0)
            
            # Create place data immediately
            place_data = {
                "name": name,
                "lat": lat,
                "lng": lng,
                "rating": rating,
                "number": user_ratings_total,
                "place_id": place_id,
                "type": place_type,
                "opening_hours": place.get('regularOpeningHours', {}).get('weekdayDescriptions', []),
               
            }
            
            # Handle photos asynchronously if they exist
            if 'photos' in place:
                photo_references = [photo.get('name') for photo in place['photos'] if photo.get('name')]
                if photo_references:
                    # Start background task for photo processing without waiting
                    task = asyncio.create_task(self.process_photos_async(photo_references, place_id, images_collection))
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                    
                                
            return name, place_data
        except Exception as e:
            print(f"Error processing place: {e}")
            return None, None

    async def process_photos_async(self, photo_references, place_id, images_collection):
        try:
            async with self.connection_semaphore:  # Limit concurrent connections
                # Check if images already exist in database
                existing_images = await images_collection.find({"place_id": place_id}).to_list(length=None)
                if existing_images:
                    return
                
                # Process photos in smaller chunks
                chunk_size = 5
                for i in range(0, len(photo_references), chunk_size):
                    chunk = photo_references[i:i + chunk_size]
                    photos = RetrievePhotos(chunk, place_id)
                    photo_responses = await photos.get_photos()
                    
                    for response in photo_responses:
                        if response is not None:
                            image_doc = {
                                "place_id": place_id,
                                "image": base64.b64encode(response.body).decode('utf-8'),
                                "content_type": "image/jpeg",
                                "timestamp": datetime.now()
                            }
                            await images_collection.insert_one(image_doc)
                    
                    # Add delay between chunks
                    if i + chunk_size < len(photo_references):
                        await asyncio.sleep(0.2)
                
        except Exception as e:
            print(f"Error in background photo processing: {e}")

    async def touristplace(self, session, place_type):
        async with self.semaphore:
            try:
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
                    
                    # Process places in smaller batches to manage connections
                    batch_size = 10
                    for i in range(0, len(results), batch_size):
                        batch = results[i:i + batch_size]
                        tasks = [self.process_single_place(place, place_type, images_collection) for place in batch]
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for name, place_data in batch_results:
                            if name and place_data:  # Only add successfully processed places
                                places[name] = place_data
                        
                        # Small delay between batches
                        if i + batch_size < len(results):
                            await asyncio.sleep(0.1)
                    
                    return places
            except Exception as e:
                print(f"Error in touristplace: {e}")
                return {}

    async def other(self, session, place_type):
        async with self.semaphore2:
            try:
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
                    
                    # Process places in batches of 5
                    batch_size = 5
                    for i in range(0, len(results), batch_size):
                        batch = results[i:i + batch_size]
                        tasks = [self.process_single_place(place, place_type, images_collection) for place in batch]
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for name, place_data in batch_results:
                            if name and place_data:  # Only add successfully processed places
                                places[name] = place_data
                        
                        # Add a small delay between batches to prevent overwhelming the system
                        if i + batch_size < len(results):
                            await asyncio.sleep(0.5)
                    
                    return places
            except Exception as e:
                print(f"Error in other: {e}")
                return {}

    async def food(self, session, place_type):
        async with self.semaphore2:
            try:
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
                    
                    # Process places in batches of 5
                    batch_size = 5
                    for i in range(0, len(results), batch_size):
                        batch = results[i:i + batch_size]
                        tasks = [self.process_single_place(place, place_type, images_collection) for place in batch]
                        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                        
                        for name, place_data in batch_results:
                            if name and place_data:  # Only add successfully processed places
                                places[name] = place_data
                        
                        # Add a small delay between batches to prevent overwhelming the system
                        if i + batch_size < len(results):
                            await asyncio.sleep(0.5)
                    
                    return places
            except Exception as e:
                print(f"Error in food: {e}")
                return {}

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
            
            # Check photo processing status after getting results
            await self.check_photo_processing_status()
            
            return final_result

    async def wait_for_background_tasks(self):
        """Wait for all background tasks to complete"""
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks)

    def duplicateremover(self, js):
        res = {}
        dict_obj = json.loads(json.dumps(js))
        res = {k: v for k, v in dict_obj.items() if k not in res}
        return res

    async def check_photo_processing_status(self):
        """Check and print the status of all photo processing tasks"""
        if not self.background_tasks:
            print("No photo processing tasks running")
            return
        
        # Check if any tasks are still running
        running_tasks = [task for task in self.background_tasks if not task.done()]
        if not running_tasks:
            print("✅ All photo processing tasks completed!")
            return
        
        # Print status of running tasks
        print(f"🔄 {len(running_tasks)} photo processing tasks still running...")

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
        self.retry_delays = [1, 2, 4, 8, 16]  # Exponential backoff delays in seconds
        self.max_retries = 3

    async def send_get_request(self, session, url):
        for retry in range(self.max_retries):
            try:
                async with self.semaphore:
                    # Add a small delay between requests to avoid rate limiting
                    await asyncio.sleep(0.2)  # 200ms delay between requests
                    
                    async with session.get(url) as photo_response:
                        if photo_response.status == 200:
                            try:
                                if photo_response.headers.get('Content-Type') == 'application/json':
                                    final_photo_url = (await photo_response.json())['photoUri']
                                    # Add delay before second request
                                    await asyncio.sleep(0.2)
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
                                if retry < self.max_retries - 1:
                                    await asyncio.sleep(self.retry_delays[retry])
                                    continue
                                return None
                        elif photo_response.status == 429:  # Too Many Requests
                            retry_after = int(photo_response.headers.get('Retry-After', self.retry_delays[retry]))
                            print(f"Rate limited. Waiting {retry_after} seconds...")
                            await asyncio.sleep(retry_after)
                            continue
                        else:
                            print(f"Error fetching photo: {photo_response.status}")
                            if retry < self.max_retries - 1:
                                await asyncio.sleep(self.retry_delays[retry])
                                continue
                            return None
            except Exception as e:
                print(f"Error in send_get_request: {e}")
                if retry < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delays[retry])
                    continue
                return None
        return None

    async def get_photos(self):
        async with aiohttp.ClientSession() as session:
            # Process photos in smaller batches
            batch_size = 10
            results = []
            
            for i in range(0, len(self.photoreferences), batch_size):
                batch = self.photoreferences[i:i + batch_size]
                tasks = [
                    self.send_get_request(
                        session,
                        f"https://places.googleapis.com/v1/{ref}/media?maxHeightPx=400&maxWidthPx=400&key=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
                    )
                    for ref in batch
                ]
                batch_results = await asyncio.gather(*tasks)
                results.extend([r for r in batch_results if r is not None])
                
                # Add delay between batches
                if i + batch_size < len(self.photoreferences):
                    await asyncio.sleep(1)  # 1 second delay between batches
            
            return results