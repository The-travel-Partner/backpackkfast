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
            fedresult = await places.getAll()
            print('test')
            print(json.dumps(fedresult, indent=4))
            print('test')
            if placefind is not None:
                dbplace = placesDBClass(cityname=self.city_name, db=self.placesdb, placetypes=self.place_types)
                findplace = await dbplace.getCity()
                finalres = await dbplace.update_json(findplace,fedresult)
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
        if 'tourist_attraction' in self.place_types:
            
            if findplace is None:
                prompt = copy.deepcopy(result['tourist_attraction'])
                finaltour = {}
                for j in prompt:
                    k = prompt[j]
                    if 'photo_refs' in k:
                        k.pop('photo_refs')
                    finaltour[j] = k
            else:
                finaltour = result['tourist_attraction']

            if await self.request.is_disconnected():
                print("Client disconnected during step 3.")
                return {"status": "Process stopped"}
            
            client = genai.Client(
                vertexai=True,
                project="backpackk",
                location="asia-south1",
            )
            from google.genai import types
            tools = [
                types.Tool(google_search=types.GoogleSearch()),
            ]

            textsi_1 = """
                               give me the response in json style like this

                              {  "Udaigarh Udaipur": {
                                       "name": "Udaigarh Udaipur",
                                       "lat": 24.5789306,
                                       "lng": 73.6827909,
                                       "rating": 4.4,
                                       "number": 1743,
                                       "place_id": "ChIJPThigWblZzkRFAHsqIB7HOk",
                                       "type": "religious"
                                   },
                                   "Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola": {
                                       "name": "Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola",
                                       "lat": 24.5801483,
                                       "lng": 73.6801341,
                                       "rating": 4,
                                       "number": 1129,
                                       "place_id": "ChIJvQBATWblZzkR6UuJd-OjSE8",
                                       "type": "religious"
                                   },
                                   "Authentic art gallery": {
                                       "name": "Authentic art gallery",
                                       "lat": 24.5853644,
                                       "lng": 73.6830902,
                                       "rating": 5,
                                       "number": 2,
                                       "place_id": "ChIJu3lJ7xLlZzkRg13xJEA01Zg",
                                       "type": "museum"
                                   }"""

            model = "gemini-2.5-flash"
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=f"""
                                                 Suggest me 20 most searched tourist attractions in {self.city_name} from this places data


                                                  Minimum rating of 4 or above
                                                  Minimum 700 reviews
                                                  Include 1 or 2 lakes if there are any
                                                  {finaltour}
                                                  Give the response in json format like this only 
                                                  "Udaigarh Udaipur": 
                                       "name": "Udaigarh Udaipur",
                                       "lat": '24.5789306',
                                       "lng": '73.6827909',
                                       "rating": '4.4',
                                       "number": '1743',
                                       "place_id": "ChIJPThigWblZzkRFAHsqIB7HOk",
                                       "type": "religious"

                                      Exclude temples, cinemas, travel agencies, amusement parks, hotels and buisness

                                      Give me 20 places minimum
                                      only give me the json structure with proper formatting, I know you can output only string, but do it in json structure
                                                 """)
                    ]
                ),
            ]
            
            generate_content_config = types.GenerateContentConfig(
                temperature = 1,
                top_p = 0.95,
                max_output_tokens = 8192,
                response_modalities = ["TEXT"],
                safety_settings = [types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="OFF"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="OFF"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="OFF"
                ),types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="OFF"
                )],
                tools = tools,
                system_instruction=[types.Part.from_text(text=textsi_1)],
            )

            response_text = ""
            for chunk in client.models.generate_content_stream(
                model = model,
                contents = contents,
                config = generate_content_config,
            ):
                if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                    continue
                response_text += chunk.text

            js = response_text
            print("Raw response from Gemini:")
            print(js)

            # Robust JSON cleaning and extraction
            def clean_and_extract_json(text):
                """
                Clean and extract JSON from Gemini response with multiple fallback strategies
                """
                try:
                    # Remove common markdown formatting
                    cleaned_text = text.replace("```json", "").replace("```", "")
                    
                    # Remove any leading/trailing whitespace
                    cleaned_text = cleaned_text.strip()
                    
                    # Check if we have any content at all
                    if not cleaned_text:
                        print("Empty response from Gemini")
                        return {}
                    
                    # Try to find JSON content between curly braces
                    import re
                    
                    # Strategy 1: Look for complete JSON object from first { to last }
                    json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            print("Strategy 1 failed, trying strategy 2...")
                    
                    # Strategy 2: Find the largest valid JSON object
                    brace_count = 0
                    start_idx = -1
                    end_idx = -1
                    
                    for i, char in enumerate(cleaned_text):
                        if char == '{':
                            if start_idx == -1:
                                start_idx = i
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0 and start_idx != -1:
                                end_idx = i
                                break
                    
                    if start_idx != -1 and end_idx != -1:
                        json_str = cleaned_text[start_idx:end_idx + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            print("Strategy 2 failed, trying strategy 3...")
                    
                    # Strategy 3: Line by line cleaning and reconstruction
                    lines = cleaned_text.split('\n')
                    cleaned_lines = []
                    in_json = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.startswith('{') or in_json:
                            in_json = True
                            # Remove any non-JSON text at the beginning of lines
                            if ':' in line or '}' in line or '{' in line or line.startswith('"'):
                                cleaned_lines.append(line)
                        if line.endswith('}') and in_json:
                            break
                    
                    if cleaned_lines:
                        reconstructed_json = '\n'.join(cleaned_lines)
                        try:
                            return json.loads(reconstructed_json)
                        except json.JSONDecodeError:
                            print("Strategy 3 failed, trying strategy 4...")
                    
                    # Strategy 4: Try to parse as-is after basic cleaning
                    try:
                        return json.loads(cleaned_text)
                    except json.JSONDecodeError:
                        print("All JSON parsing strategies failed")
                        return {}
                        
                except Exception as e:
                    print(f"Error in JSON cleaning: {e}")
                    return {}
            
            # Use the robust JSON extraction
            finaljson = clean_and_extract_json(js)
            
            if finaljson:
                print("Successfully parsed JSON:")
                print(json.dumps(finaljson, indent=4))
            else:
                print("Failed to extract valid JSON from response, using original tourist attractions")
                finaljson = {}
                
            print('Original result data:')
            print(json.dumps(result,indent=4))
            # Debug prints to understand the data
            print("Keys in result['tourist_attraction']:", list(result['tourist_attraction'].keys()))
            print("Keys in finaljson:", list(finaljson.keys()))
            
            # Check if there's a mismatch between keys
            # Try to match by name instead of exact key match
            filtered_json = {}
            for key, value in result['tourist_attraction'].items():
                place_name = value.get('name', '')
                # Check if this place name exists in finaljson
                for final_key in finaljson.keys():
                    if place_name.lower() in final_key.lower() or final_key.lower() in place_name.lower():
                        filtered_json[key] = value
                        break
            
            
            # If still empty, use the original data
            if not filtered_json:
                print("No matches found, using original data")
                filtered_json = result['tourist_attraction']
            
            result['tourist_attraction'] = filtered_json
            print(json.dumps(result['tourist_attraction'], indent=4))

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
                s['reviews'] = [result[types][item]['reviews']]
                
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

        print(lat_lng_string)

        def split_into_batches(lat_lng_string, batch_size=5):

            lat_lon_list = lat_lng_string.split('|')

            batches = ['|'.join(lat_lon_list[i:i + batch_size]) for i in range(0, len(lat_lon_list), batch_size)]

            return batches

        batches = split_into_batches(lat_lng_string, batch_size=5)

        # Print the batches
        poly = []
        dist = []
        
        import asyncio
        import random
        
        async def rate_limited_distance_matrix(client, start, end, max_retries=3):
            """
            Rate limited distance matrix call with exponential backoff
            """
            base_delay = 2  # Base delay in seconds
            max_delay = 60  # Maximum delay in seconds
            
            for attempt in range(max_retries):
                try:
                    # Add random jitter to prevent thundering herd
                    if attempt > 0:
                        jitter = random.uniform(0.1, 0.5)
                        delay = min(base_delay * (2 ** attempt) + jitter, max_delay)
                        print(f"Rate limit hit, waiting {delay:.2f} seconds before retry {attempt + 1}...")
                        await asyncio.sleep(delay)
                    
                    print(f"Attempting distance matrix call for batch: {end[:50]}...")
                    distance_matrix = client.routing.distance_matrix(start, end)
                    print(f"Successfully got distance matrix for batch")
                    return distance_matrix
                    
                except Exception as e:
                    if "429" in str(e) or "rate limit" in str(e).lower():
                        print(f"Rate limit hit on attempt {attempt + 1}: {e}")
                        if attempt == max_retries - 1:
                            print(f"Max retries reached, skipping this batch")
                            return None
                        continue
                    else:
                        print(f"Non-rate-limit error: {e}")
                        return None
            
            return None
        
        # Process batches with rate limiting
        for i, end_batch in enumerate(batches):
            start = "26.9854865,75.8513454"
            end = end_batch
            
            print(f"Processing batch {i + 1}/{len(batches)}")
            
            # Add delay between batches to respect rate limits
            if i > 0:
                batch_delay = 1.5  # 1.5 seconds between batches
                print(f"Waiting {batch_delay} seconds between batches...")
                await asyncio.sleep(batch_delay)

            distance_matrix = await rate_limited_distance_matrix(client, start, end)
            
            if distance_matrix is None:
                print(f"Skipping batch {i + 1} due to persistent errors")
                # Add default values for this batch
                batch_size = len(end.split('|'))
                for _ in range(batch_size):
                    poly.append(None)
                    dist.append(0)  # Default distance
                continue

            for row in distance_matrix.get('rows', []):
                for element in row.get('elements', []):
                    distance = element.get('distance')
                    polys = element.get('polyline')

                    if polys is not None:
                        poly.append(polys)
                    else:
                        poly.append(None)
                        
                    if distance is not None:
                        dist.append(distance)
                    else:
                        dist.append(0)  # Default distance if not available

        df_sorted['distance'] = dist
        df_sorted['path'] = poly

        return df_sorted,placesdata


