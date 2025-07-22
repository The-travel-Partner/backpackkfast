import json
import time
from datetime import datetime, timedelta

from py_olamaps.OlaMaps import OlaMaps
import googlemaps
import google.generativeai as genai
from googlemaps.distance_matrix import distance_matrix
from vertexai.generative_models import GenerativeModel, Part, Tool
import vertexai.preview.generative_models as generative_models
import vertexai
import pandas as pd
from tripgen.asyncclass import place
import sys
import copy
import re
from tripgen.placesDBClass import placesDBClass
from tripgen.bestplacesModel import bestPlacesModel
import heapq
import asyncio
from math import radians, sin, cos, sqrt, atan2

class TripCreator:
    def __init__(self, request, city_name, place_types, no_of_days, useremail, placesdb, weekday, travel_schedule, df_sorted=None):
        self.request = request
        self.city_name = city_name
        self.place_types = place_types
        self.no_of_days = no_of_days
        self.useremail = useremail
        self.placesdb = placesdb
        self.weekday = weekday
        self.travel_schedule = travel_schedule
        self.df_sorted = df_sorted

    async def create_trip(self):

        df_sorted = self.df_sorted

        client = OlaMaps(
            api_key="hlN2QjeqJ9hBh2tq1AVvw6O50ilZX4gCfPtbgx6j",
            client_id="41b89c52-b013-41a3-b2a4-f0ca47d68d8b",
            client_secret="IvSD5AfH21DFYh0qHgoqRmLNZIFtyXRM"
        )

        northeast = bounds['northeast']
        southwest = bounds['southwest']
        print(northeast)

        northeast = (northeast['lat'], northeast['lng'])
        southwest = (southwest['lat'], southwest['lng'])
        print('Northeast coordinates:', northeast)
        print('Southwest coordinates:', southwest)

        places = place(placetypes=self.place_types, northeast=northeast, southwest=southwest)
        result = await places.getAll()
        print(json.dumps(result['tourist_attraction'], indent=4))
        prompt = copy.deepcopy(result['tourist_attraction'])
        finaltour = {}
        for j in prompt:
            k = prompt[j]
            k.pop('photos')
            finaltour[j] = k


        vertexai.init(project='backpackk', location='asia-south1')
        tools = [
            Tool.from_google_search_retrieval(
                google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval(disable_attribution=False)
            ),
        ]

        textsi_1 = """
               give me the response in json style like this

                "Udaigarh Udaipur": {
                       "name": "Udaigarh Udaipur",
                       "lat": 24.5789306,
                       "lng": 73.6827909,
                       "rating": 4.4,
                       "number": 1743,
                       "place_id": \"ChIJPThigWblZzkRFAHsqIB7HOk",
                       "type": "religious"
                   },
                   "Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola": {
                       "name": \"Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola\",
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

        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 1,
            "top_p": 0.95,

        }
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        model = GenerativeModel(
            "gemini-1.5-pro-preview-0409",
            tools=tools,
            system_instruction=[textsi_1]
        )
        print(len(df_sorted))
        def read_csv(csv):
            places = []
            food = []
            night_life = []
            for index, row in csv.iterrows():
                print(row)

                if row['type'] == 'restaurant' or row['type'] == 'vegetarian_restaurant':
                    food.append({
                        'Name': row['Name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'rating': row['rating'],
                        'number': row['number'],
                        'place_id': row['place_id'],
                        'type': row['type'],
                        'weight_avg': float(row['weighted_avg']),
                        'distance': float(row['distance']),
                        'opening_hours': row['opening_hours']
                    })
                elif row['type'] == 'night_club' or row['type'] == 'bar':
                    night_life.append({
                        'Name': row['Name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'rating': row['rating'],
                        'number': row['number'],
                        'place_id': row['place_id'],
                        'type': row['type'],
                        'weight_avg': float(row['weighted_avg']),
                        'distance': float(row['distance']),
                        'opening_hours': row['opening_hours']
                    })
                else:
                    places.append({
                        'Name': row['Name'],
                        'lat': row['lat'],
                        'lng': row['lng'],
                        'rating': row['rating'],
                        'number': row['number'],
                        'place_id': row['place_id'],
                        'type': row['type'],
                        'weight_avg': float(row['weighted_avg']),
                        'distance': float(row['distance']),
                        'opening_hours': row['opening_hours']
                    })
        
            return places, food, night_life

        convo = model.start_chat()
        resp = {}
        ans = {}

        text = ""
        for i in range(1, len(self.place_types)):
            text = text + " " + self.place_types[i]
        places = 20
        output = 0
        js = convo.send_message([f"""
                                  Suggest me 20 most searched tourist attractions in {self.city_name} from this places data

                                  exclude {text}
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
                   ,
                   "Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola": 
                       "name": "Hotel Sarovar - A Boutique Lake Facing Hotel On Lake Pichola",
                       "lat": '24.5801483',
                       "lng": '73.6801341',
                       "rating": '4',
                       "number": '1129',
                       "place_id": "ChIJvQBATWblZzkR6UuJd-OjSE8",
                       "type": "religious"

                      Exclude temples, cinemas, travel agencies, amusement parks, hotels and buisness

                      Give me 20 places minimum
                      only give me the json with proper formatting
                                  """], generation_config=generation_config
                                ).to_dict()['candidates'][0]['content']['parts'][0]['text']

        print(js)
        convo.history.clear()

        js = js.replace("json", "")
        js = js.replace("```", "")
        finaljson = json.loads(js)
        filtered_json = {key: result['tourist_attraction'][key] for key in result['tourist_attraction'] if key in finaljson}
        result['tourist_attraction'] = filtered_json
        print(json.dumps(result['tourist_attraction'],indent=4))

        df = pd.DataFrame()
        for types in self.place_types:
            for item in result[types]:
                j=[]
                s = pd.DataFrame(columns=['Name', 'lat', 'lng', 'rating', 'number', 'place_id', 'type','photos'])
                s['Name'] = [result[types][item]['name']]
                s['lat'] = [result[types][item]['lat']]
                s['lng'] = [result[types][item]['lng']]
                s['rating'] = [result[types][item]['rating']]
                s['number'] = [result[types][item]['number']]
                s['place_id'] = [result[types][item]['place_id']]
                s['type'] = [result[types][item]['type']]
                for photo in result[types][item]['photos']:
                    print(photo['name'])
                    j.append(photo['name'])
                print(j)

                s['photos'].loc[0] = j



        def next_day(current_day):
            # List of days in order
            days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

            # Make the input case-insensitive
            current_day = current_day.capitalize()

            if current_day not in days_of_week:
                return "Invalid day entered."

            # Find the index of the current day and calculate the next day
            current_index = days_of_week.index(current_day)
            next_index = (current_index + 1) % len(days_of_week)  # Use modulo to wrap around to Monday
            return days_of_week[next_index]
        def create_itinerary(places, num_days, travel_schedule, food, night_life, place_types):
            def calculate_score(place, current_distance):
                weight = place['weight_avg']
                if weight == 0:
                    print("weight is 0")
                    weight =1
                return (place['distance'] / weight) - current_distance

            itinerary = {}
            current_distance = 0
            remaining_places = places[1:]
            # print(len(remaining_places))
            slots = []
            if (travel_schedule == "Leisure"):
                start_time = datetime.strptime("14:30", "%H:%M")
                end_time = datetime.strptime("20:00", "%H:%M")
                slots.append("10:00-13:00")
                split = (end_time - start_time).total_seconds() / 60 / 2
                for i in range(2):
                    end_time = start_time + timedelta(minutes=split)
                    slots.append(start_time.strftime("%H:%M") + "-" + end_time.strftime("%H:%M"))
                    start_time = end_time


            elif (travel_schedule == "Explorer"):
                start_morning = datetime.strptime("09:00", "%H:%M")
                end_morning = datetime.strptime("13:00", "%H:%M")
                split_morning = (end_morning - start_morning).total_seconds() / 60 / 2
                start_afternoon = datetime.strptime("14:30", "%H:%M")
                end_afternoon = datetime.strptime("20:00", "%H:%M")
                split_afternoon = (end_afternoon - start_afternoon).total_seconds() / 60 / 2

                for i in range(2):
                    morning = start_morning + timedelta(minutes=split_morning)
                    slots.append(start_morning.strftime("%H:%M") + "-" + morning.strftime("%H:%M"))
                    start_morning = morning
                for i in range(2):
                    afternoon = start_afternoon + timedelta(minutes=split_afternoon)
                    slots.append(start_afternoon.strftime("%H:%M") + "-" + afternoon.strftime("%H:%M"))
                    start_afternoon = afternoon


            elif (travel_schedule == "Adventurer"):
                start_morning = datetime.strptime("09:00", "%H:%M")
                end_morning = datetime.strptime("13:00", "%H:%M")
                split_morning = (end_morning - start_morning).total_seconds() / 60 / 2
                start_afternoon = datetime.strptime("14:30", "%H:%M")
                end_afternoon = datetime.strptime("20:00", "%H:%M")
                split_afternoon = (end_afternoon - start_afternoon).total_seconds() / 60 / 3

                for i in range(2):
                    morning = start_morning + timedelta(minutes=split_morning)
                    slots.append(start_morning.strftime("%H:%M") + "-" + morning.strftime("%H:%M"))
                    start_morning = morning
                for i in range(3):
                    afternoon = start_afternoon + timedelta(minutes=split_afternoon)
                    slots.append(start_afternoon.strftime("%H:%M") + "-" + afternoon.strftime("%H:%M"))
                    start_afternoon = afternoon
            weekday = self.weekday
            for day in range(1, num_days + 1):

                print(weekday)

                current_remainingplaces = copy.deepcopy(remaining_places)
                current_food = copy.deepcopy(food)
                current_night = copy.deepcopy(night_life)


                day_places = []
                useless = []
                lunch_restro = []
                dinner_restro = []
                night = []
         
                i = 0
                try:
                    while len(day_places) < len(slots) :
                        print("working")
                        i = len(day_places)


                        pq = [(calculate_score(place, current_distance), idx, place) for idx, place in enumerate(current_remainingplaces)]

                        heapq.heapify(pq)

                        _, _, best_place = heapq.heappop(pq)

                        if ((travel_schedule == "Leisure" and i == 0) or (
                                travel_schedule == "Leisure" and i == 2)):
                            if i == 0:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)
                                lunch_restro.append(best_food)
                                current_food.remove(best_food)
                                food.remove(best_food)
                            elif i == 2:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)
                                dinner_restro.append(best_food)
                                current_distance = best_food['distance']
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_score = [(calculate_score(night, current_distance), idx, night) for idx, night in enumerate(current_night)]
                                    heapq.heapify(night_score)
                                    _, _, best_night = heapq.heappop(night_score)
                                    night.append(best_night)
                                    current_night.remove(best_night)
                                    night_life.remove(best_night)
                                current_food.remove(best_food)
                                food.remove(best_food)

                        elif ((travel_schedule == "Explorer" and i == 1) or (
                                travel_schedule == "Explorer" and i == 3)):
                            if i == 1:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)

                                lunch_restro.append(best_food)
                                current_food.remove(best_food)
                                food.remove(best_food)
                            elif i == 3:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)

                                dinner_restro.append(best_food)
                                current_distance = best_food['distance']
                                if ('night_club' in place_types) or ('bar' in place_types):
                                    night_score = [(calculate_score(night, current_distance), idx, night) for idx, night in enumerate(current_night)]
                                    heapq.heapify(night_score)
                                    _, _, best_night = heapq.heappop(night_score)
                                    night.append(best_night)
                                    current_night.remove(best_night)
                                    night_life.remove(best_night)
                                current_food.remove(best_food)
                                food.remove(best_food)

                        elif ((travel_schedule == "Adventurer" and i == 1) or (
                                travel_schedule == "Adventurer" and i == 4)):
                            if i == 1:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)

                                lunch_restro.append(best_food)
                                current_food.remove(best_food)
                                food.remove(best_food)
                            elif i == 4:

                                food_score = [(calculate_score(restro, current_distance), idx, restro) for idx, restro in enumerate(current_food)]
                                heapq.heapify(food_score)
                                _, _, best_food = heapq.heappop(food_score)

                                dinner_restro.append(best_food)
                                current_distance = best_food['distance']
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_score = [(calculate_score(night, current_distance), idx, night) for idx, night in enumerate(current_night)]
                                    heapq.heapify(night_score)
                                    _, _, best_night = heapq.heappop(night_score)
                                    night.append(best_night)
                                    current_night.remove(best_night)
                                    night_life.remove(best_night)
                                current_food.remove(best_food)
                                food.remove(best_food)

                        day_places.append(best_place)
                        
                        if best_place in current_remainingplaces:
                            current_remainingplaces.remove(best_place)
                        if best_place in remaining_places:
                            remaining_places.remove(best_place)
                        current_distance = best_place['distance']

                        for j in range(len(useless)):
                            current_remainingplaces.append(useless[j])
                except Exception as e:
                    print(e)
                    return int(day) - 2
                    

                itinerary[f"Day {day}"] = json.loads(pd.DataFrame(day_places).to_json())
                itinerary[f"Day {day}"].update({"Weekday": weekday})
                itinerary[f"Day {day}"].update({"Lunch": json.loads(pd.DataFrame(lunch_restro).to_json())})
                itinerary[f"Day {day}"].update({"Dinner": json.loads(pd.DataFrame(dinner_restro).to_json())})
                if 'night_club' in place_types or 'bar' in place_types:
                    itinerary[f"Day {day}"].update({"Night": json.loads(pd.DataFrame(night).to_json())})
            # print(len(remaining_places))
                weekday = next_day(str(weekday))
            return itinerary

        places, food, night_life = read_csv(df_sorted)
        
       

        day_wise = create_itinerary(places=places, num_days=self.no_of_days, travel_schedule=self.travel_schedule, food=food, night_life=night_life, place_types=self.place_types)

        print(day_wise)
        if type(day_wise) is int:
            
            return day_wise

        else:


            async def get_distance_matrix(start_lat, start_lng, end_lat, end_lng, max_retries=3):
                start_point = f"{start_lat},{start_lng}"
                end_point = f"{end_lat},{end_lng}"
                
                # Retry logic with exponential backoff
                retry_delays = [1, 2, 4]  # Delays in seconds
                
                for attempt in range(max_retries):
                    try:
                        # Validate coordinates
                        if not all(isinstance(x, (int, float)) for x in [start_lat, start_lng, end_lat, end_lng]):
                            raise ValueError("Invalid coordinate values")
                        
                        if not (-90 <= start_lat <= 90 and -90 <= end_lat <= 90 and
                               -180 <= start_lng <= 180 and -180 <= end_lng <= 180):
                            raise ValueError("Coordinates out of valid range")
                        
                        # Get distance matrix - note: this is synchronous
                        distance_matrix = client.routing.distance_matrix(start_point, end_point)
                        
                        # Validate response
                        if not distance_matrix or not isinstance(distance_matrix, dict):
                            raise ValueError("Invalid distance matrix response")
                        
                        # Check if we got valid distance and duration
                        if 'rows' not in distance_matrix or not distance_matrix['rows']:
                            raise ValueError("No distance data in response")
                        
                        first_row = distance_matrix['rows'][0]
                        if 'elements' not in first_row or not first_row['elements']:
                            raise ValueError("No route elements in response")
                        
                        first_element = first_row['elements'][0]
                        if 'status' not in first_element:
                            raise ValueError("No status in route element")
                        
                        # Check if the route is valid
                        if first_element['status'] != 'OK':
                            # If route not found, return a fallback distance
                            if first_element['status'] == 'NOT_FOUND':
                                # Calculate rough distance using Haversine formula
                                R = 6371  # Earth's radius in kilometers
                                
                                lat1, lon1 = radians(start_lat), radians(start_lng)
                                lat2, lon2 = radians(end_lat), radians(end_lng)
                                
                                dlat = lat2 - lat1
                                dlon = lon2 - lon1
                                
                                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                                c = 2 * atan2(sqrt(a), sqrt(1-a))
                                distance = R * c
                                
                                # Create fallback response
                                return {
                                    'rows': [{
                                        'elements': [{
                                            'distance': {'value': int(distance * 1000), 'text': f"{distance:.1f} km"},
                                            'duration': int(distance * 200),  # Return as integer
                                            'status': 'FALLBACK'
                                        }]
                                    }]
                                }
                            
                            raise ValueError(f"Route status: {first_element['status']}")
                        
                        return distance_matrix
                        
                    except Exception as e:
                        if attempt < max_retries - 1:
                            # Log the error and wait before retrying
                            print(f"Attempt {attempt + 1} failed: {str(e)}")
                            await asyncio.sleep(retry_delays[attempt])
                            continue
                        else:
                            # On final attempt, return fallback with error info
                            print(f"All attempts failed: {str(e)}")
                            return {
                                'rows': [{
                                    'elements': [{
                                        'distance': {'value': 0, 'text': 'Unknown'},
                                        'duration': 0,  # Return as integer
                                        'status': 'ERROR',
                                        'error': str(e)
                                    }]
                                }]
                            }


            async def durations(travel_schedule, place_types):
                async def travel_dur(start_lat, start_lng, end_lat, end_lng):
                    distance_matrix = await get_distance_matrix(start_lat, start_lng, end_lat, end_lng)

                    if 'rows' in distance_matrix:
                        duration = distance_matrix['rows'][0]['elements'][0]['duration']
                        # Handle both dictionary and integer formats
                        if isinstance(duration, dict):
                            duration_seconds = duration['value']
                        else:
                            duration_seconds = duration
                        duration_minutes = duration_seconds // 60
                        return duration_minutes
                    else:
                        print(f"    Error fetching distance matrix: {distance_matrix}")
                        raise Exception(f"Error fetching distance matrix: {distance_matrix}")

                durations = []
                for day in day_wise:
                    duration = []
                    if travel_schedule == 'Leisure':
                        for i in range(2):
                            start_lat = day_wise[day]['lat'][str(i)]
                            start_lng = day_wise[day]['lng'][str(i)]
                            end_lat = day_wise[day]['lat'][str(i + 1)]
                            end_lng = day_wise[day]['lng'][str(i + 1)]

                            if i == 0:
                                lunch_lat = day_wise[day]['Lunch']['lat'][str(0)]
                                lunch_lng = day_wise[day]['Lunch']['lng'][str(0)]
                                print(lunch_lat, lunch_lng)

                                duration.append(await travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    await travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))

                            if i == 1:
                                duration.append(await travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(await travel_dur(start_lat, start_lng, dinner_lat, dinner_lng))
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(await travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))

                    elif travel_schedule == 'Explorer':
                        print(travel_schedule)
                        for i in range(3):
                            start_lat = day_wise[day]['lat'][str(i)]
                            start_lng = day_wise[day]['lng'][str(i)]
                            end_lat = day_wise[day]['lat'][str(i + 1)]
                            end_lng = day_wise[day]['lng'][str(i + 1)]
                            if i == 1:
                                lunch_lat = day_wise[day]['Lunch']['lat'][str(0)]
                                lunch_lng = day_wise[day]['Lunch']['lng'][str(0)]
                                duration.append(await travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    await travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))

                            elif i == 2:
                                duration.append(await travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(await travel_dur(start_lat, start_lng, dinner_lat, dinner_lng))
                                if ('night_club' in place_types) or ('bar' in place_types):
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(await travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))
                            else:
                                duration.append(await travel_dur(start_lat, start_lng, end_lat, end_lng))

                    elif travel_schedule == "Adventurer":
                        for i in range(4):
                            start_lat = day_wise[day]['lat'][str(i)]
                            start_lng = day_wise[day]['lng'][str(i)]
                            end_lat = day_wise[day]['lat'][str(i + 1)]
                            end_lng = day_wise[day]['lng'][str(i + 1)]
                            if i == 1:
                                lunch_lat = day_wise[day]['Lunch']['lat'][str(0)]
                                lunch_lng = day_wise[day]['Lunch']['lng'][str(0)]
                                duration.append(await travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    await travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))

                            elif i == 3:
                                duration.append(await travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(await travel_dur(end_lat, end_lng, dinner_lat, dinner_lng))
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(await travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))
                            else:
                                duration.append(await travel_dur(start_lat, start_lng, end_lat, end_lng))
                    durations.append(duration)

                return durations

            dur = await durations(self.travel_schedule, self.place_types)

            def convert_minutes(minutes):
                hours = minutes // 60
                remaining_minutes = minutes % 60
                return f"{int(hours):02}:{int(remaining_minutes):02}"

            def create_dynamic_slots_with_fixed_lunch(day_wise, start_time_str, end_time_str, travel_durations, lunch_start_str, lunch_end_str, dinner_end_str, travel_schedule):
                """
                Create dynamic time slots for each day with robust error handling and validation.
                
                Args:
                    day_wise: Dictionary containing day-wise place data
                    start_time_str: Start time of the day (e.g., "09:00")
                    end_time_str: End time of main activities (e.g., "20:00")
                    travel_durations: List of travel durations for each day
                    lunch_start_str: Lunch start time (e.g., "13:00")
                    lunch_end_str: Lunch end time (e.g., "14:30")
                    dinner_end_str: Dinner end time (e.g., "21:30")
                    travel_schedule: Schedule type ("Leisure", "Explorer", "Adventurer")
                
                Returns:
                    Updated day_wise dictionary with slot information
                """
                
                # Schedule configuration with validation
                schedule_config = {
                    "Explorer": {"morning": (0, 2), "afternoon": (3, 5), "min_slot_duration": 30},
                    "Adventurer": {"morning": (0, 2), "afternoon": (3, 6), "min_slot_duration": 25},
                    "Leisure": {"morning": (0, 1), "afternoon": (1, 3), "min_slot_duration": 45}
                }
                
                # Validate travel schedule
                if travel_schedule not in schedule_config:
                    print(f"Warning: Unknown travel schedule '{travel_schedule}', defaulting to Explorer")
                    travel_schedule = "Explorer"
                
                config = schedule_config[travel_schedule]
                morning_start, morning_end = config["morning"]
                afternoon_start, afternoon_end = config["afternoon"]
                min_slot_duration = config["min_slot_duration"]
                
                def safe_time_parse(time_str, default_time="09:00"):
                    """Safely parse time string with fallback"""
                    try:
                        return datetime.strptime(time_str, "%H:%M")
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid time format '{time_str}', using default '{default_time}'")
                        return datetime.strptime(default_time, "%H:%M")
                
                def validate_time_sequence(times_dict):
                    """Validate that times are in logical sequence"""
                    time_order = ['start', 'lunch_start', 'lunch_end', 'end', 'dinner_end']
                    for i in range(len(time_order) - 1):
                        if times_dict[time_order[i]] >= times_dict[time_order[i + 1]]:
                            print(f"Warning: Time sequence issue - {time_order[i]} >= {time_order[i + 1]}")
                            return False
                    return True
                
                def calculate_optimal_slot_duration(available_time, num_slots, travel_time, min_duration):
                    """Calculate optimal slot duration with constraints"""
                    if num_slots <= 0:
                        return min_duration
                    
                    raw_duration = (available_time - travel_time) / num_slots
                    
                    # Ensure minimum duration
                    if raw_duration < min_duration:
                        print(f"Warning: Calculated slot duration ({raw_duration:.1f}min) below minimum ({min_duration}min)")
                        return min_duration
                    
                    return max(raw_duration, min_duration)
                
                a = 0
                for day in day_wise:
                    try:
                        # Validate travel durations
                        if a >= len(dur):
                            print(f"Warning: No travel duration data for day {a + 1}, skipping")
                            continue
                            
                        travel_durations = dur[a]
                        
                        # Validate travel durations list
                        if not travel_durations or len(travel_durations) < afternoon_end:
                            print(f"Warning: Insufficient travel duration data for day {a + 1}")
                            # Pad with default values
                            while len(travel_durations) < afternoon_end:
                                travel_durations.append(15)  # Default 15 minutes
                        
                        # Parse and validate all times
                        times = {
                            'start': safe_time_parse(start_time_str, "09:00"),
                            'end': safe_time_parse(end_time_str, "20:00"),
                            'lunch_start': safe_time_parse(lunch_start_str, "13:00"),
                            'lunch_end': safe_time_parse(lunch_end_str, "14:30"),
                            'dinner_end': safe_time_parse(dinner_end_str, "21:30")
                        }
                        
                        # Validate time sequence
                        if not validate_time_sequence(times):
                            print(f"Adjusting times for day {a + 1} due to sequence issues")
                            # Auto-correct common issues
                            if times['lunch_start'] <= times['start']:
                                times['lunch_start'] = times['start'] + timedelta(hours=3)
                            if times['lunch_end'] <= times['lunch_start']:
                                times['lunch_end'] = times['lunch_start'] + timedelta(minutes=90)
                            if times['end'] <= times['lunch_end']:
                                times['end'] = times['lunch_end'] + timedelta(hours=4)
                        
                        # Calculate available time periods
                        morning_time = (times['lunch_start'] - times['start']).total_seconds() / 60
                        afternoon_time = (times['end'] - times['lunch_end']).total_seconds() / 60
                        
                        # Calculate travel times
                        try:
                            travel_duration_morning = sum(travel_durations[x] for x in range(morning_start, morning_end) if x < len(travel_durations))
                            travel_duration_afternoon = sum(travel_durations[x] for x in range(afternoon_start, min(afternoon_end, len(travel_durations))))
                        except (IndexError, TypeError) as e:
                            print(f"Warning: Error calculating travel durations for day {a + 1}: {e}")
                            travel_duration_morning = (morning_end - morning_start) * 15
                            travel_duration_afternoon = (afternoon_end - afternoon_start) * 15
                        
                        # Calculate slot durations
                        morning_slots = morning_end - morning_start
                        afternoon_slots = afternoon_end - afternoon_start
                        
                        morning_slot_duration = calculate_optimal_slot_duration(
                            morning_time, morning_slots, travel_duration_morning, min_slot_duration
                        )
                        
                        afternoon_slot_duration = calculate_optimal_slot_duration(
                            afternoon_time, afternoon_slots, travel_duration_afternoon, min_slot_duration
                        )
                        
                        # Build slots
                        slot = {}
                        slot_start = times['start']
                        
                        # Morning slots
                        for i in range(morning_start, morning_end):
                            if i >= len(travel_durations):
                                print(f"Warning: Missing travel duration for slot {i}, using default")
                                travel_time = 15
                            else:
                                travel_time = max(0, travel_durations[i])  # Ensure non-negative
                            
                            slot_end = slot_start + timedelta(minutes=morning_slot_duration)
                            
                            slot[f"{i}"] = {
                                "Start": slot_start.strftime("%H:%M"),
                                "End": slot_end.strftime("%H:%M"),
                                "Slot Duration": convert_minutes(morning_slot_duration),
                                "Travel Duration": travel_time,
                                "Type": "Morning Activity"
                            }
                            
                            slot_start = slot_end + timedelta(minutes=travel_time)
                            
                            # Add lunch after last morning slot
                            if i == morning_end - 1:
                                lunch_duration = (times['lunch_end'] - times['lunch_start']).total_seconds() / 60
                                slot['Lunch'] = {
                                    "Start": times['lunch_start'].strftime("%H:%M"),
                                    "End": times['lunch_end'].strftime("%H:%M"),
                                    "Slot Duration": convert_minutes(lunch_duration),
                                    "Travel Duration": travel_time,
                                    "Type": "Meal"
                                }
                                slot_start = times['lunch_end'] + timedelta(minutes=travel_time)
                        
                        # Afternoon slots
                        last_place_end_time = slot_start
                        
                        for i in range(afternoon_start, afternoon_end):
                            if i >= len(travel_durations):
                                travel_time = 15
                            else:
                                travel_time = max(0, travel_durations[i])
                            
                            slot_end = slot_start + timedelta(minutes=afternoon_slot_duration)
                            
                            slot[f"{i}"] = {
                                "Start": slot_start.strftime("%H:%M"),
                                "End": slot_end.strftime("%H:%M"),
                                "Slot Duration": convert_minutes(afternoon_slot_duration),
                                "Travel Duration": travel_time,
                                "Type": "Afternoon Activity"
                            }
                            
                            slot_start = slot_end + timedelta(minutes=travel_time)
                            last_place_end_time = slot_start
                        
                        # Dynamic dinner timing
                        last_travel_time = travel_durations[-1] if travel_durations else 15
                        actual_dinner_start = max(
                            times['end'], 
                            last_place_end_time - timedelta(minutes=last_travel_time)
                        )
                        
                        # Ensure dinner has reasonable duration
                        dinner_duration = (times['dinner_end'] - actual_dinner_start).total_seconds() / 60
                        if dinner_duration < 30:  # Minimum 30 minutes for dinner
                            print(f"Warning: Dinner duration too short ({dinner_duration:.1f}min), extending dinner end time")
                            times['dinner_end'] = actual_dinner_start + timedelta(minutes=90)
                            dinner_duration = 90
                        
                        slot["Dinner"] = {
                            "Start": actual_dinner_start.strftime("%H:%M"),
                            "End": times['dinner_end'].strftime("%H:%M"),
                            "Slot Duration": convert_minutes(dinner_duration),
                            "Travel Duration": last_travel_time,
                            "Type": "Meal"
                        }
                        
                        # Night activities (if applicable)
                        if "Night" in day_wise[day]:
                            night_start = times['dinner_end'] + timedelta(minutes=last_travel_time)
                            night_end = night_start + timedelta(minutes=90)
                            slot["Night"] = {
                                "Start": night_start.strftime("%H:%M"),
                                "End": night_end.strftime("%H:%M"),
                                "Slot Duration": convert_minutes(90),
                                "Travel Duration": 0,
                                "Type": "Night Activity"
                            }
                        
                        # Convert to DataFrame and add to day_wise
                        try:
                            df = pd.DataFrame(slot).T
                            finalslot = json.loads(df.to_json())
                            day_wise[day].update({"Slots": finalslot})
                        except Exception as e:
                            print(f"Error creating slots DataFrame for day {a + 1}: {e}")
                            # Create a simple fallback slot structure
                            day_wise[day].update({"Slots": {"Error": "Failed to create detailed slots"}})
                        
                    except Exception as e:
                        print(f"Error processing day {a + 1}: {e}")
                        # Add error information to the day
                        day_wise[day].update({"Slots": {"Error": f"Slot creation failed: {str(e)}"}})
                    
                    a += 1
                
                return day_wise

            main_start = "09:00"
            main_end = "20:00"

            lunch_start = "13:00"
            lunch_end = "14:30"
            dinner_end = "21:30"
            night_start = "22:00"
            itinerary = create_dynamic_slots_with_fixed_lunch(day_wise, main_start, main_end, dur, lunch_start, lunch_end,
                                                         dinner_end, self.travel_schedule)
            print(json.dumps(itinerary,indent=4))
            return itinerary





