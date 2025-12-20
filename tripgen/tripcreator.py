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
from tripgen.placesRetrieve import placesRetrieve
import sys
import copy
import re
from tripgen.placesDBClass import placesDBClass
from tripgen.bestplacesModel import bestPlacesModel
import heapq
import asyncio
from math import radians, sin, cos, sqrt, atan2

class TripCreator:
    def __init__(self, city_name, place_types, no_of_days, placesdb, weekday, travel_schedule, useremail, request=None):
      
        self.city_name = city_name
        self.place_types = place_types
        self.no_of_days = no_of_days
        
        self.placesdb = placesdb
        self.weekday = weekday
        self.travel_schedule = travel_schedule
        self.useremail = useremail
        self.request = request

    async def create_trip(self):

        # Use placesRetrieve to get places data instead of duplicating logic
        places_retriever = placesRetrieve(
            request=self.request,
            city_name=self.city_name,
            place_types=self.place_types,
            useremail=self.useremail,
            placesdb=self.placesdb
        )
        
        df_sorted, placesdata = await places_retriever.getplaces()

        client = OlaMaps(
            api_key="hlN2QjeqJ9hBh2tq1AVvw6O50ilZX4gCfPtbgx6j",
            client_id="41b89c52-b013-41a3-b2a4-f0ca47d68d8b",
            client_secret="IvSD5AfH21DFYh0qHgoqRmLNZIFtyXRM"
        )



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
        def create_itinerary(places, num_days, travel_schedule, place_types):
            def calculate_score(place, current_distance):
                weight = place['weight_avg']
                if weight == 0:
                    weight = 1
                return (place['distance'] / weight) - current_distance

            itinerary = {}
            current_distance = 0
            remaining_places = places.copy()  # Use all places, don't skip first
            
            # Define slot configurations for different travel schedules
            schedule_config = {
                "Leisure": {"total_slots": 3, "lunch_after": 1, "dinner_after": 2},
                "Explorer": {"total_slots": 4, "lunch_after": 1, "dinner_after": 3}, 
                "Adventurer": {"total_slots": 5, "lunch_after": 1, "dinner_after": 4}
            }
            
            config = schedule_config.get(travel_schedule, schedule_config["Explorer"])
            
            weekday = self.weekday
            for day in range(1, num_days + 1):
                print(f"Creating itinerary for {weekday}")

                # Create continuous sequence of all places for the day
                continuous_sequence = []
                current_distance_temp = current_distance
                
                # Get attractions, restaurants, and nightlife from ALL remaining places
                attractions = [p for p in remaining_places if p['type'] not in ['restaurant', 'vegetarian_restaurant', 'night_club', 'bar']]
                restaurants = [p for p in remaining_places if p['type'] in ['restaurant', 'vegetarian_restaurant']]
                nightlife = [p for p in remaining_places if p['type'] in ['night_club', 'bar']]
                
                print(f"Day {day}: Found {len(attractions)} attractions, {len(restaurants)} restaurants, {len(nightlife)} nightlife")
                
                try:
                    # Add places for each slot with meals integrated
                    for slot in range(config["total_slots"]):
                        # Add attraction for this slot
                        if attractions:
                            pq = [(calculate_score(place, current_distance_temp), idx, place) 
                                  for idx, place in enumerate(attractions)]
                            heapq.heapify(pq)
                            _, _, best_place = heapq.heappop(pq)
                            
                            continuous_sequence.append(best_place)
                            attractions.remove(best_place)
                            remaining_places.remove(best_place)
                            current_distance_temp = best_place['distance']
                            print(f"Added attraction: {best_place['Name']}")
                        
                        # Add lunch after specified slot
                        if slot == config["lunch_after"] and restaurants:
                            food_score = [(calculate_score(restro, current_distance_temp), idx, restro) 
                                         for idx, restro in enumerate(restaurants)]
                            heapq.heapify(food_score)
                            _, _, best_lunch = heapq.heappop(food_score)
                            
                            continuous_sequence.append(best_lunch)
                            restaurants.remove(best_lunch)
                            remaining_places.remove(best_lunch)
                            current_distance_temp = best_lunch['distance']
                            print(f"Added restaurant: {best_lunch['Name']}")
                        
                        # Add dinner after specified slot
                        if slot == config["dinner_after"] and restaurants:
                            food_score = [(calculate_score(restro, current_distance_temp), idx, restro) 
                                         for idx, restro in enumerate(restaurants)]
                            heapq.heapify(food_score)
                            _, _, best_dinner = heapq.heappop(food_score)
                            
                            continuous_sequence.append(best_dinner)
                            restaurants.remove(best_dinner)
                            remaining_places.remove(best_dinner)
                            current_distance_temp = best_dinner['distance']
                            print(f"Added dinner: {best_dinner['Name']}")
                            
                            # Add night venue if applicable
                            if ('night_club' in place_types or 'bar' in place_types) and nightlife:
                                night_score = [(calculate_score(night, current_distance_temp), idx, night) 
                                              for idx, night in enumerate(nightlife)]
                                heapq.heapify(night_score)
                                _, _, best_night = heapq.heappop(night_score)
                                
                                continuous_sequence.append(best_night)
                                nightlife.remove(best_night)
                                remaining_places.remove(best_night)
                                current_distance_temp = best_night['distance']
                                print(f"Added nightlife: {best_night['Name']}")
                    
                    print(f"Day {day} total sequence: {len(continuous_sequence)} places")
                    
                    # Convert continuous sequence to the required JSON structure
                    if continuous_sequence:
                        day_data = {}
                        
                        # Initialize all field dictionaries
                        fields = ['Name', 'lat', 'lng', 'rating', 'number', 'place_id', 'type', 
                                 'weight_avg', 'distance', 'opening_hours']
                        for field in fields:
                            day_data[field] = {}
                        
                        # Populate with continuous numbering
                        for idx, place in enumerate(continuous_sequence):
                            day_data['Name'][str(idx)] = place['Name']
                            day_data['lat'][str(idx)] = place['lat']
                            day_data['lng'][str(idx)] = place['lng']
                            day_data['rating'][str(idx)] = place['rating']
                            day_data['number'][str(idx)] = place['number']
                            day_data['place_id'][str(idx)] = place['place_id']
                            day_data['type'][str(idx)] = place['type']
                            day_data['weight_avg'][str(idx)] = place['weight_avg']
                            day_data['distance'][str(idx)] = place['distance']
                            day_data['opening_hours'][str(idx)] = place['opening_hours']
                        
                        # Add weekday
                        day_data['Weekday'] = weekday
                        
                        itinerary[f"Day {day}"] = day_data
                    
                except Exception as e:
                    print(f"Error creating itinerary for day {day}: {e}")
                    return int(day) - 1
                
                # Update current distance for next day
                if continuous_sequence:
                    current_distance = continuous_sequence[-1]['distance']
                
                weekday = next_day(str(weekday))
            
            return itinerary

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
                    'weight_avg': float(row['weighted_avg']),
                    'distance': float(row['distance']),
                    'opening_hours': row['opening_hours']
                })
            return places

        places = read_csv(df_sorted)
        
        # Debug: Check what place types we actually have
        place_type_counts = {}
        for place in places:
            place_type = place['type']
            place_type_counts[place_type] = place_type_counts.get(place_type, 0) + 1
        
        print("Available place types and counts:")
        for ptype, count in place_type_counts.items():
            print(f"  {ptype}: {count}")
        
        print(f"Requested place_types: {self.place_types}")
        
        # Check if we have restaurants and nightlife
        restaurants = [p for p in places if p['type'] in ['restaurant', 'vegetarian_restaurant']]
        nightlife = [p for p in places if p['type'] in ['night_club', 'bar']]
        
        print(f"Found {len(restaurants)} restaurants and {len(nightlife)} nightlife venues")
        
        if len(restaurants) == 0:
            print("WARNING: No restaurants found in data!")
        if len(nightlife) == 0:
            print("WARNING: No nightlife venues found in data!")
        
       

        day_wise = create_itinerary(places=places, num_days=self.no_of_days, travel_schedule=self.travel_schedule, place_types=self.place_types)

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
                    day_data = day_wise[day]
                    
                    # Get total number of places for this day
                    total_places = len(day_data.get('Name', {}))
                    
                    # Calculate travel durations between consecutive places
                    for i in range(total_places - 1):
                        try:
                            start_lat = day_data['lat'][str(i)]
                            start_lng = day_data['lng'][str(i)]
                            end_lat = day_data['lat'][str(i + 1)]
                            end_lng = day_data['lng'][str(i + 1)]
                            
                            travel_time = await travel_dur(start_lat=start_lat, start_lng=start_lng, 
                                                         end_lat=end_lat, end_lng=end_lng)
                            duration.append(travel_time)
                            
                        except (KeyError, ValueError) as e:
                            print(f"Error calculating duration between places {i} and {i+1}: {e}")
                            duration.append(15)  # Default 15 minutes
                    
                    durations.append(duration)

                return durations

            dur = await durations(self.travel_schedule, self.place_types)

            def convert_minutes(minutes):
                hours = minutes // 60
                remaining_minutes = minutes % 60
                return f"{int(hours):02}:{int(remaining_minutes):02}"

            def create_dynamic_slots_with_fixed_lunch(day_wise, start_time_str, end_time_str, travel_durations, lunch_start_str, lunch_end_str, dinner_end_str, travel_schedule):
                """
                Create dynamic time slots for each day with the new continuous structure.
                """
                
                def safe_time_parse(time_str, default_time="09:00"):
                    """Safely parse time string with fallback"""
                    try:
                        return datetime.strptime(time_str, "%H:%M")
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid time format '{time_str}', using default '{default_time}'")
                        return datetime.strptime(default_time, "%H:%M")

                def identify_place_type(place_type):
                    """Identify if a place is lunch, dinner, or night venue"""
                    if place_type in ['restaurant', 'vegetarian_restaurant']:
                        return 'meal'
                    elif place_type in ['night_club', 'bar']:
                        return 'night'
                    else:
                        return 'attraction'

                # Standard time configurations
                times = {
                    'start': safe_time_parse(start_time_str, "09:00"),
                    'lunch_start': safe_time_parse(lunch_start_str, "13:00"),
                    'lunch_end': safe_time_parse(lunch_end_str, "14:30"),
                    'dinner_start': safe_time_parse("20:00", "20:00"),
                    'dinner_end': safe_time_parse(dinner_end_str, "21:30"),
                    'night_start': safe_time_parse("22:00", "22:00")
                }

                a = 0
                for day in day_wise:
                    try:
                        day_data = day_wise[day]
                        
                        # Get travel durations for this day
                        if a >= len(dur):
                            print(f"Warning: No travel duration data for day {a + 1}, using defaults")
                            travel_durations = [15] * (len(day_data.get('Name', {})) - 1)
                        else:
                            travel_durations = dur[a]
                        
                        # Get total places for this day
                        total_places = len(day_data.get('Name', {}))
                        
                        if total_places == 0:
                            continue
                        
                        # Create slots for each place
                        slots = {}
                        current_time = times['start']
                        
                        lunch_assigned = False
                        dinner_assigned = False
                        
                        for i in range(total_places):
                            place_type = day_data.get('type', {}).get(str(i), 'attraction')
                            place_category = identify_place_type(place_type)
                            
                            # Determine slot duration based on place type and schedule
                            if place_category == 'meal':
                                if not lunch_assigned and current_time < times['dinner_start']:
                                    # This is lunch
                                    slot_duration_minutes = (times['lunch_end'] - times['lunch_start']).total_seconds() / 60
                                    if current_time < times['lunch_start']:
                                        current_time = times['lunch_start']
                                    lunch_assigned = True
                                    activity_type = "Lunch"
                                elif not dinner_assigned:
                                    # This is dinner
                                    slot_duration_minutes = (times['dinner_end'] - times['dinner_start']).total_seconds() / 60
                                    if current_time < times['dinner_start']:
                                        current_time = times['dinner_start']
                                    dinner_assigned = True
                                    activity_type = "Dinner"
                                else:
                                    # Additional meal venue
                                    slot_duration_minutes = 60
                                    activity_type = "Meal"
                            elif place_category == 'night':
                                # Night venue
                                slot_duration_minutes = 90
                                if current_time < times['night_start']:
                                    current_time = times['night_start']
                                activity_type = "Night Activity"
                            else:
                                # Regular attraction - base duration on travel schedule
                                if travel_schedule == "Leisure":
                                    slot_duration_minutes = 120  # 2 hours
                                elif travel_schedule == "Explorer":
                                    slot_duration_minutes = 90   # 1.5 hours
                                else:  # Adventurer
                                    slot_duration_minutes = 75   # 1.25 hours
                                activity_type = "Attraction"
                            
                            # Calculate end time
                            slot_end = current_time + timedelta(minutes=slot_duration_minutes)
                            
                            # Get travel time to next place
                            if i < len(travel_durations):
                                travel_time = max(0, travel_durations[i])
                            else:
                                travel_time = 0
                            
                            # Store slot information
                            slots[str(i)] = {
                                "Start": current_time.strftime("%H:%M"),
                                "End": slot_end.strftime("%H:%M"),
                                "Slot Duration": convert_minutes(slot_duration_minutes),
                                "Travel Duration": travel_time,
                                "Type": activity_type
                            }
                            
                            # Update current time for next slot
                            current_time = slot_end + timedelta(minutes=travel_time)
                        
                        # Add slots to day data
                        try:
                            # Convert slots to the expected format
                            slot_df_data = {
                                "Start": {k: v["Start"] for k, v in slots.items()},
                                "End": {k: v["End"] for k, v in slots.items()},
                                "Slot Duration": {k: v["Slot Duration"] for k, v in slots.items()},
                                "Travel Duration": {k: v["Travel Duration"] for k, v in slots.items()},
                                "Type": {k: v["Type"] for k, v in slots.items()}
                            }
                            
                            day_wise[day]["Slots"] = slot_df_data
                            
                        except Exception as e:
                            print(f"Error creating slots for day {a + 1}: {e}")
                            day_wise[day]["Slots"] = {"Error": f"Slot creation failed: {str(e)}"}
                        
                    except Exception as e:
                        print(f"Error processing day {a + 1}: {e}")
                        day_wise[day]["Slots"] = {"Error": f"Day processing failed: {str(e)}"}
                    
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





