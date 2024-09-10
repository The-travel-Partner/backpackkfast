import json
import time
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
from tripgen.placesDBClass import placesDBClass
import heapq
class TripCreator:
    def __init__(self, request, city_name, place_types, no_of_days, useremail, placesdb):
        self.request = request
        self.city_name = city_name
        self.place_types = place_types
        self.no_of_days = no_of_days
        self.useremail = useremail
        self.placesdb = placesdb


    async def create_trip(self):
        gmaps = googlemaps.Client(key='AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk')
        genai.configure(api_key='AIzaSyDqWSMHmOR-4kmyc8GWH9IGjrgHHsh2dJ8')
        print(self.city_name)
        print(self.no_of_days)
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
            if month==int(timelist[0]) and year==int(timelist[1]):
                dbplace = placesDBClass(cityname=self.city_name, db=self.placesdb, placetypes=self.place_types)
                findplace = await dbplace.getCity()
                print(json.dumps(findplace, indent=4))
            else:
                findplace=None
        print('checkpoint')
        print(findplace)
        result = {}
        fedresult={}
        if findplace is None:

            new_types = ['tourist_attraction', 'museum','zoo','night_club','bar','hindu_temple','church','mosque']
            places = place(placetypes=new_types, northeast=northeast, southwest=southwest)
            if await self.request.is_disconnected():
                print("Client disconnected during step 3.")
                return {"status": "Process stopped"}
            fedresult = await places.getAll()

            for types in self.place_types:
                result[f"{types}"] =copy.deepcopy(fedresult[types])
        else:
            result = findplace


        if fedresult !={}:
            placesdata = copy.deepcopy(fedresult)
        else:
            placesdata = None
        if 'tourist_attraction' in self.place_types:


            if findplace is None:
                prompt = copy.deepcopy(result['tourist_attraction'])
                finaltour = {}
                for j in prompt:
                    k = prompt[j]
                    k.pop('photos')
                    finaltour[j] = k
            else:
                finaltour= result['tourist_attraction']

            if await self.request.is_disconnected():
                print("Client disconnected during step 3.")
                return {"status": "Process stopped"}
            vertexai.init(project='backpackk-cloud', location='asia-south1')
            tools = [
                Tool.from_google_search_retrieval(
                    google_search_retrieval=generative_models.grounding.GoogleSearchRetrieval(disable_attribution=False)
                ),
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
                "gemini-1.5-flash-001",
                tools=tools,
                system_instruction=textsi_1
            )

            convo = model.start_chat(response_validation=False)
            resp = {}
            ans = {}


            if await self.request.is_disconnected():
                print("Client disconnected during step 3.")
                return {"status": "Process stopped"}
            js = convo.send_message([f"""
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
                          only give me the json with proper formatting
                                     """],generation_config=generation_config,safety_settings=safety_settings
                                    ).to_dict()['candidates'][0]['content']['parts'][0]['text']

            print(js)
            convo.history.clear()

            js = js.replace("json", "")
            js = js.replace("```", "")
            finaljson = json.loads(js)

            filtered_json = {key: result['tourist_attraction'][key] for key in result['tourist_attraction'] if
                             key in finaljson}
            result['tourist_attraction'] = filtered_json
            print(json.dumps(result['tourist_attraction'], indent=4))

        df = pd.DataFrame()
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        for types in self.place_types:
            for item in result[types]:
                j = []
                s = pd.DataFrame(columns=['Name', 'lat', 'lng', 'rating', 'number', 'place_id', 'type','photos'])
                s['Name'] = [result[types][item]['name']]
                s['lat'] = [result[types][item]['lat']]
                s['lng'] = [result[types][item]['lng']]
                s['rating'] = [result[types][item]['rating']]
                s['number'] = [result[types][item]['number']]
                s['place_id'] = [result[types][item]['place_id']]
                s['type'] = [result[types][item]['type']]
               
                for photo in result[types][item]['photos']:

                    j.append(photo)


                s['photos'].loc[0] = j
                df = pd.concat([df, s], axis=0, ignore_index=True)
        weights = [0.3, 0.7]
        df['weighted_avg'] = df.apply(lambda x: (float(x['rating']) * weights[0] + float(x['number']) * weights[1]),
                                      axis=1)
        df_sorted = df.sort_values(by='weighted_avg', ascending=False)
        df_sorted = df_sorted.drop_duplicates(subset='Name', keep='first', inplace=False)
        df_sorted.to_csv('bigdata.csv')
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}


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
        for end_batch in batches:
            start = "26.9854865,75.8513454"
            end = end_batch

            distance_matrix = client.routing.distance_matrix(start, end)



            for row in distance_matrix.get('rows', []):
                for element in row.get('elements', []):
                    distance = element.get('distance')
                    polys = element.get('polyline')
                    if polys is not None:
                        poly.append(polys)
                    if distance is not None:
                        dist.append(distance)

        df_sorted['distance'] = dist
        df_sorted['path'] = poly

        def read_csv(csv):
            places = []

            for index, row in csv.iterrows():
                places.append({
                    'Name': row['Name'],
                    'lat': row['lat'],
                    'lng': row['lng'],
                    'rating': row['rating'],
                    'number': row['number'],
                    'place_id': row['place_id'],
                    'type': row['type'],

                    'photos': row['photos'],
                    'weight_avg': float(row['weighted_avg']),
                    'distance': float(row['distance']),

                    'path': row['path']

                })

            return places

        def create_itinerary(places, num_days):
            def calculate_score(place, current_distance):
                return (place['distance'] / place['weight_avg']) - current_distance

            itinerary = {}
            current_distance = 0
            remaining_places = places[1:]

            for day in range(1, num_days + 1):

                day_places = []
                while len(day_places) < 4 and remaining_places:
                    pq = [(calculate_score(place, current_distance), place) for place in remaining_places]
                    heapq.heapify(pq)

                    _, best_place = heapq.heappop(pq)
                    day_places.append(best_place)
                    remaining_places.remove(best_place)
                    current_distance = best_place['distance']

                itinerary[f"Day {day}"] = json.loads(pd.DataFrame(day_places).to_json())

            return itinerary

        places = read_csv(df_sorted)
        itinerary = create_itinerary(places, self.no_of_days)
        print(json.dumps(placesdata, indent=4))


        return itinerary, placesdata



