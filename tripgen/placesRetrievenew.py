import json
import time
from datetime import datetime, timedelta

from py_olamaps.OlaMaps import OlaMaps
import googlemaps
from google import genai
from google.genai import types
import base64
from googlemaps.distance_matrix import distance_matrix
import pandas as pd
from tripgen.asyncclass import place
import sys
import copy

import re
from tripgen.placesDBClass import placesDBClass
from tripgen.bestplacesModel import bestPlacesModel
from bson.objectid import ObjectId

class placesRetrieve:
    def __init__(self,request, city_name, place_types, useremail, placesdb):
        self.request=request
        self.city_name = city_name
        self.place_types = place_types
        self.useremail = useremail
        self.placesdb = placesdb

    async def getplaces(self):
        gmaps = googlemaps.Client(key='AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk')
        
        print(self.city_name)

        geocode_result = gmaps.geocode(self.city_name)

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
        placesCollection = self.placesdb['placesdata']
        placefind = await placesCollection.find_one({"city_name": self.city_name})
        findplace = None
        typecheck = None
        if placefind is not None:
            timestamp = placefind['timestamp']
            timelist = timestamp.split(',')
            year = int(time.localtime().tm_year)
            month = int(time.localtime().tm_mon)
            if month == int(timelist[0]) and year == int(timelist[1]):
                dbplace = placesDBClass(cityname=self.city_name, db=self.placesdb, placetypes=self.place_types)
                findplace = await dbplace.getCity()
              
            else:
                findplace = None
        print(findplace)
        print('checkpoint')
 
        result = {}
        fedresult = {}
        if findplace is None:

            new_types = ['tourist_attraction', 'museum', 'zoo', 'night_club', 'bar', 'hindu_temple', 'church', 'mosque',
                         'restaurant', 'vegetarian_restaurant']
            places = place(placetypes=new_types, northeast=northeast, southwest=southwest, placesdb=self.placesdb)
            if await self.request.is_disconnected():
                print("Client disconnected during step 3.")
                return {"status": "Process stopped"}
            print('test')
            fedresult = await places.getAll()
            
            print(json.dumps(fedresult, indent=4))
            print('test')
            if placefind is not None:
                dbplace = placesDBClass(cityname=self.city_name, db=self.placesdb, placetypes=self.place_types)
                findplace = await dbplace.getCity()
                finalres = dbplace.update_json(findplace,fedresult)
                timestamp = f"{str(time.localtime().tm_mon)},{str(time.localtime().tm_year)}"
                places = {'city_name': self.city_name, "timestamp": timestamp, 'places': finalres}
                filter_criteria = {"city_name": self.city_name}
                collection = self.placesdb['placesdata']
                result = await collection.replace_one(filter_criteria, fedresult)
            for types in self.place_types:
                result[f"{types}"] = copy.deepcopy(fedresult[types])
        else:
            result = findplace

        if fedresult != {}:
            placesdata = copy.deepcopy(fedresult)
        else:
            placesdata = None
    
           
        df = pd.DataFrame()
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        for types in self.place_types:
            for item in result[types]:
                s = pd.DataFrame(columns=['Name', 'lat', 'lng', 'rating', 'number', 'place_id', 'type', 'opening_hours'])
                s['Name'] = [result[types][item]['name']]
                s['lat'] = [result[types][item]['lat']]
                s['lng'] = [result[types][item]['lng']]
                s['rating'] = [result[types][item]['rating']]
                s['number'] = [result[types][item]['number']]
                s['place_id'] = [result[types][item]['place_id']]
                s['type'] = [result[types][item]['type']]
                s['opening_hours'] = [result[types][item]['opening_hours']]
                
                # We're no longer fetching photos
                
                df = pd.concat([df, s], axis=0, ignore_index=True)
        weights = [0.3, 0.7]
        df['weighted_avg'] = df.apply(lambda x: (float(x['rating']) * weights[0] + float(x['number']) * weights[1]),
                                      axis=1)
        df_sorted = df.sort_values(by='weighted_avg', ascending=False)
        df_sorted = df_sorted.drop_duplicates(subset='Name', keep='first', inplace=False)

        client = OlaMaps(
            api_key="hlN2QjeqJ9hBh2tq1AVvw6O50ilZX4gCfPtbgx6j",
            client_id="41b89c52-b013-41a3-b2a4-f0ca47d68d8b",
            client_secret="IvSD5AfH21DFYh0qHgoqRmLNZIFtyXRM"
        )
        latitudes = df_sorted['lat']
        longitudes = df_sorted['lng']

        lat_lng_string = '|'.join(f"{lat},{lon}" for lat, lon in zip(latitudes, longitudes))

        def split_into_batches(lat_lng_string, batch_size=5):
            lat_lon_list = lat_lng_string.split('|')
            batches = ['|'.join(lat_lon_list[i:i + batch_size]) for i in range(0, len(lat_lon_list), batch_size)]
            return batches

        batches = split_into_batches(lat_lng_string, batch_size=5)

        async def process_batch(end_batch):
            start = "26.9854865,75.8513454"
            end = end_batch
            try:
                # Run the synchronous API call in a separate thread
                distance_matrix = await asyncio.to_thread(
                    client.routing.distance_matrix,
                    start,
                    end
                )
                batch_poly = []
                batch_dist = []
                
                for row in distance_matrix.get('rows', []):
                    for element in row.get('elements', []):
                        distance = element.get('distance')
                        polys = element.get('polyline')
                        
                        if polys is not None:
                            batch_poly.append(polys)
                        if distance is not None:
                            batch_dist.append(distance)
                            
                return batch_poly, batch_dist
            except Exception as e:
                print(f"Error processing batch: {e}")
                return [], []

        # Process batches concurrently
        import asyncio
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        
        # Combine results
        poly = []
        dist = []
        for batch_poly, batch_dist in results:
            poly.extend(batch_poly)
            dist.extend(batch_dist)

        df_sorted['distance'] = dist
        
        df_sorted.to_csv('df_sorted.csv', index=False)
        return df_sorted,placesdata


