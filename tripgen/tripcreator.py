import json
import time

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

class TripCreator:
    def __init__(self,request, city_name, place_types, no_of_days):
        self.request=request
        self.city_name = city_name
        self.place_types = place_types
        self.no_of_days = no_of_days

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

        places = place(placetypes=self.place_types, northeast=northeast, southwest=southwest)
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        result = await places.getAll()
        print(json.dumps(result['tourist_attraction'], indent=4))

        prompt = copy.deepcopy(result['tourist_attraction'])
        finaltour = {}
        for j in prompt:
            k = prompt[j]
            k.pop('photos')
            finaltour[j] = k

        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
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

        convo = model.start_chat()
        resp = {}
        ans = {}

        text = ""
        for i in range(1, len(self.place_types)):
            text = text + " " + self.place_types[i]
        places = 20
        output = 0
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
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

        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        print(js)
        convo.history.clear()

        js = js.replace("json", "")
        js = js.replace("```", "")
        finaljson = json.loads(js)
        filtered_json = {key: result['tourist_attraction'][key] for key in result['tourist_attraction'] if key in finaljson}
        result['tourist_attraction'] = filtered_json
        print(json.dumps(result['tourist_attraction'],indent=4))

        df = pd.DataFrame()
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
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
        gmaps = googlemaps.Client(key='AAIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk')


        def compute_distance_matrix(data):
            df_dist_matrix = pd.DataFrame(index=data.index, columns=data.index)

            chunk_size = 10
            for i in range(0, len(data), chunk_size):
                origins = [(loc['lat'], loc['lng']) for _, loc in data.iloc[i:i + chunk_size].iterrows()]
                for j in range(0, len(data), chunk_size):
                    destinations = [(loc['lat'], loc['lng']) for _, loc in data.iloc[j:j + chunk_size].iterrows()]

                    try:
                        result = distance_matrix(gmaps, origins, destinations, mode='driving')
                        for k, row in enumerate(result['rows']):
                            for l, element in enumerate(row['elements']):
                                df_dist_matrix.at[data.index[i + k], data.index[j + l]] = element['distance']['value']
                        time.sleep(1)
                    except googlemaps.exceptions.ApiError as e:
                        print(f"Error calculating distance matrix: {e}")
                        return pd.DataFrame()

            return df_dist_matrix

        n = int(self.no_of_days)
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        df_distances = compute_distance_matrix(df_sorted)
        print(df_distances)
        threshold = 50
        df_sorted = df_sorted[df_sorted['weighted_avg'] >= threshold]

        def divide_places(n, data, df_distances):
            sorted_df = data.sort_values(by='weighted_avg', ascending=False)

            total_places = len(sorted_df)
            places_per_day = min(total_places // n, 4)  # Maximum 4 places per day

            days = []
            for i in range(n):
                day = pd.DataFrame()
                current_location = None
                for j in range(places_per_day):
                    if len(sorted_df) == 0:
                        break
                    if current_location is None:
                        next_location = sorted_df.iloc[0]
                        day = pd.concat([day, next_location.to_frame().T], ignore_index=True)
                        sorted_df = sorted_df.drop(next_location.name)
                        current_location = next_location
                    else:
                        distances = df_distances.loc[current_location.name]
                        combined_scores = distances[sorted_df.index].rank(ascending=True) + \
                                          sorted_df['weighted_avg'].rank(ascending=False)
                        best_index = combined_scores.idxmin()
                        next_location = sorted_df.loc[best_index]
                        day = pd.concat([day, next_location.to_frame().T], ignore_index=True)
                        sorted_df = sorted_df.drop(next_location.name)
                        current_location = next_location

                days.append(day)

            return days
        if await self.request.is_disconnected():
            print("Client disconnected during step 3.")
            return {"status": "Process stopped"}
        days = divide_places(n, df_sorted, df_distances)
        day_wise = {}
        for i, day in enumerate(days,1):
                print(f"Day {i}:")
                day_wise[f"Day {i}:"] = json.loads(day.to_json())
                print(day.to_json())
                print(day)
                print()
        return day_wise