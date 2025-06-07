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

        def is_subslot_available(main_start, main_end, sub_start, sub_end):

            # Convert time strings to datetime objects
            main_start_time = datetime.strptime(main_start, "%I:%M%p")
            main_end_time = datetime.strptime(main_end, "%I:%M%p")
            sub_start_time = datetime.strptime(sub_start, "%H:%M")
            sub_end_time = datetime.strptime(sub_end, "%H:%M")

            # Check if the sub-slot is within the main time range
            if (main_start_time <= sub_start_time and sub_end_time <= main_end_time):
                return True
            else:
                return False

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
                for curplace in current_remainingplaces[:]:  # Create a copy of the list to iterate
                    if not curplace['opening_hours']:  # Check if opening hours is empty
                        current_remainingplaces.remove(curplace)
                        continue
                        
                    for opening in curplace['opening_hours']:
                        if weekday in opening:
                            if "Closed" in str(opening):
                                current_remainingplaces.remove(curplace)
                print("before",len(current_food))
                for restro in current_food[:]:  # Create a copy of the list to iterate
                    if not restro['opening_hours']:  # Check if opening hours is empty
                        current_food.remove(restro)
                        continue
                        
                    for opening in restro['opening_hours']:
                        if weekday in opening:
                            if 'Closed' in str(opening):
                                current_food.remove(restro)
                print("after",len(current_food))
                for night in current_night[:]:  # Create a copy of the list to iterate
                    if not night['opening_hours']:  # Check if opening hours is empty
                        current_night.remove(night)
                        continue
                        
                    for opening in night['opening_hours']:
                        if weekday in str(opening):
                            if 'Closed' in weekday:
                                current_night.remove(night)


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
                        current_weekday = [element for element in best_place['opening_hours'] if weekday in element]
                        if current_weekday:
                                # Replace Unicode spaces with regular spaces
                                cleaned_weekday = current_weekday[0].replace('\u202f', ' ').replace('\u2009', ' ')
                                # Replace the en dash with a regular hyphen
                                cleaned_weekday = cleaned_weekday.replace('–', '-')
                                current_weekday = [cleaned_weekday]
                                print(f"Cleaned weekday: {current_weekday}")
                        fin = current_weekday[0].replace(" ","")
                        time_pattern = r'(\d{1,2}:\d{2}(?:AM|PM)|\d{1,2}:\d{2})-(\d{1,2}:\d{2}(?:AM|PM)|\d{1,2}:\d{2})'
                        match = False
                        matches = re.findall(time_pattern, fin)
                        print("start and end",matches)
                        main_start=""
                        main_end =""
                        for start_time, end_time in matches:

                            if (("AM" in start_time or "PM" in start_time) and (
                                    "AM" in end_time or "PM" in end_time)):
                                match = True
                                main_start = start_time
                                main_end = end_time
                            elif 'Open' in fin:
                                match = True
                                main_start = "00:00"
                                main_end = "23:59"
                            else:
                                match = False
                        
                        print(match)

                        if match:



                            sub_start = slots[i].split("-")[0]
                            sub_end = slots[i].split("-")[1]
                            res = is_subslot_available(main_start=main_start, main_end=main_end, sub_start=sub_start,
                                                        sub_end=sub_end)
                            print("result",res)
                            if res:

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

                            else:
                                if best_place in current_remainingplaces:
                                    current_remainingplaces.remove(best_place)
                        else:
                            if best_place in current_remainingplaces:
                                    current_remainingplaces.remove(best_place)
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


            def get_distance_matrix(start_lat, start_lng, end_lat, end_lng):
                start_point = f"{start_lat},{start_lng}"
                end_point = f"{end_lat},{end_lng}"
                try:
                    distance_matrix = client.routing.distance_matrix(start_point, end_point)
                    return distance_matrix
                except Exception as e:
                    return f"Error getting distance matrix: {str(e)}"


            def durations(travel_schedule, place_types):
                def travel_dur(start_lat, start_lng, end_lat, end_lng):
                    distance_matrix = get_distance_matrix(start_lat, start_lng, end_lat, end_lng)

                    if 'rows' in distance_matrix:

                        duration_seconds = distance_matrix['rows'][0]['elements'][0]['duration']
                        duration_minutes = duration_seconds // 60
                        return duration_minutes
                    else:
                        print(f"    Error fetching distance matrix: {distance_matrix}")
                        Exception(f"Error fetching distance matrix: {distance_matrix}")

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

                                duration.append(travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))

                            if i == 1:
                                duration.append(travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(travel_dur(start_lat, start_lng, dinner_lat, dinner_lng))
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))




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
                                duration.append(travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))


                            elif i == 2:
                                duration.append(travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(travel_dur(start_lat, start_lng, dinner_lat, dinner_lng))
                                if ('night_club' in place_types) or ('bar' in place_types):
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))
                            else:
                                duration.append(travel_dur(start_lat, start_lng, end_lat, end_lng))


                    elif travel_schedule == "Adventurer":
                        for i in range(4):
                            start_lat = day_wise[day]['lat'][str(i)]
                            start_lng = day_wise[day]['lng'][str(i)]
                            end_lat = day_wise[day]['lat'][str(i + 1)]
                            end_lng = day_wise[day]['lng'][str(i + 1)]
                            if i == 1:
                                lunch_lat = day_wise[day]['Lunch']['lat'][str(0)]
                                lunch_lng = day_wise[day]['Lunch']['lng'][str(0)]
                                duration.append(travel_dur(start_lat=start_lat, start_lng=start_lng, end_lat=lunch_lat,
                                                           end_lng=lunch_lng))
                                duration.append(
                                    travel_dur(start_lat=lunch_lat, start_lng=lunch_lng, end_lat=end_lat, end_lng=end_lng))


                            elif i == 3:
                                duration.append(travel_dur(start_lat, start_lng, end_lat, end_lng))
                                start_lat = day_wise[day]['lat'][str(i + 1)]
                                start_lng = day_wise[day]['lng'][str(i + 1)]
                                dinner_lat = day_wise[day]['Dinner']['lat'][str(0)]
                                dinner_lng = day_wise[day]['Dinner']['lng'][str(0)]
                                duration.append(travel_dur(end_lat, end_lng, dinner_lat, dinner_lng))
                                if 'night_club' in place_types or 'bar' in place_types:
                                    night_lat = day_wise[day]['Night']['lat'][str(0)]
                                    night_lng = day_wise[day]['Night']['lng'][str(0)]
                                    duration.append(travel_dur(dinner_lat, dinner_lng, night_lat, night_lng))
                            else:
                                duration.append(travel_dur(start_lat, start_lng, end_lat, end_lng))
                    durations.append(duration)

                return durations

            dur = durations(self.travel_schedule, self.place_types)

            def convert_minutes(minutes):
                hours = minutes // 60
                remaining_minutes = minutes % 60
                return f"{int(hours):02}:{int(remaining_minutes):02}"

            def create_dynamic_slots_with_fixed_lunch(day_wise, start_time_str, end_time_str, travel_durations, lunch_start_str, lunch_end_str, dinner_end_str, travel_schedule):
                a = 0
                for day in day_wise:

                    travel_durations = dur[a]
                    if travel_schedule == "Explorer":
                        morning_start = 0
                        morning_end = 2
                        afternoon_start = 3
                        afternoon_end = 5
                    elif travel_schedule == "Adventurer":
                        morning_start = 0
                        morning_end = 2
                        afternoon_start = 3
                        afternoon_end = 6
                    elif travel_schedule == "Leisure":
                        morning_start = 0
                        morning_end = 1
                        afternoon_start = 1
                        afternoon_end = 3
                    # Convert start, end, and lunch times from strings to datetime objects
                    start_time = datetime.strptime(start_time_str, "%H:%M")
                    end_time = datetime.strptime(end_time_str, "%H:%M")
                    lunch_start = datetime.strptime(lunch_start_str, "%H:%M")
                    lunch_end = datetime.strptime(lunch_end_str, "%H:%M")

                    # Calculate total available time before lunch and after lunch
                    morning_time = (lunch_start - start_time).total_seconds() / 60  # Time before lunch in minutes
                    afternoon_time = (end_time - lunch_end).total_seconds() / 60  # Time after lunch in minutes

                    travel_duration_morning = sum(travel_durations[x] for x in range(morning_start, morning_end))

                    travel_duration_afternoon = sum(
                        travel_durations[x] for x in range(afternoon_start, len(travel_durations)))

                    total_travel_morning = morning_time - travel_duration_morning

                    if travel_schedule == 'Leisure':
                        morning_slot_duration = total_travel_morning
                    else:
                        morning_slot_duration = total_travel_morning / 2

                    total_travel_afternoon = afternoon_time - travel_duration_afternoon

                    if travel_schedule == 'Adventurer':
                        afternoon_slot_duration = total_travel_afternoon / 3
                    else:
                        afternoon_slot_duration = total_travel_afternoon / 2

                    slot = {}
                    slot_start = start_time

                    for i in range(morning_start, morning_end):

                        slot_end = slot_start + timedelta(minutes=morning_slot_duration)

                        slot[f"{i}"] = {"Start": slot_start.strftime("%H:%M"), "End": slot_end.strftime("%H:%M"),
                                        "Slot Duration": convert_minutes(morning_slot_duration),
                                        "Travel Duration": travel_durations[i]}

                        slot_start = slot_end + timedelta(minutes=travel_durations[i])
                        if i == morning_end - 1:
                            slot['Lunch'] = {"Start": lunch_start_str, "End": lunch_end_str,
                                             "Slot Duration": (lunch_end - lunch_start).total_seconds() / 60,
                                             "Travel Duration": travel_durations[i]}
                            slot_start = lunch_end + timedelta(minutes=travel_durations[i])

                    for i in range(afternoon_start, afternoon_end):
                        slot_end = slot_start + timedelta(minutes=afternoon_slot_duration)

                        slot[f"{i}"] = {"Start": slot_start.strftime("%H:%M"), "End": slot_end.strftime("%H:%M"),
                                        "Slot Duration": convert_minutes(afternoon_slot_duration),
                                        "Travel Duration": travel_durations[i]}

                        slot_start = slot_end + timedelta(minutes=travel_durations[i])

                    dinner_slot_duration = (datetime.strptime(dinner_end_str, "%H:%M") - end_time).total_seconds() / 60
                    slot["Dinner"] = {"Start": end_time_str, "End": dinner_end_str, "Slot Duration": dinner_slot_duration,
                                      "Travel Duration": travel_durations[-1:][0]}
                    if "Night" in day_wise[day]:
                        slot_start = datetime.strptime(dinner_end_str, "%H:%M") + timedelta(
                            minutes=travel_durations[-1:][0])
                        slot_end = slot_start + timedelta(minutes=90)
                        slot["Night"] = {"Start": slot_start.strftime("%H:%M"), "End": slot_end.strftime("%H:%M"),
                                         "Slot Duration": (slot_end - slot_start).total_seconds() / 60}
                    df = pd.DataFrame(slot).T
                    finalslot = json.loads(df.to_json())
                    day_wise[day].update({"Slots": finalslot})
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





