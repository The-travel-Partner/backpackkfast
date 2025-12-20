"""
Trip Time Recalculator Module

This module provides functionality to recalculate time slots for rearranged trip data
received from the trip/modify endpoint. It handles both the legacy JSON structure with
nested dictionaries for places, meals, and the new continuous structure where all places
are in a single sequence with type classification.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from math import radians, sin, cos, sqrt, atan2

# Try to import OlaMaps, fallback to None if not available
try:
    from py_olamaps.OlaMaps import OlaMaps
    HAS_OLA_MAPS = True
except ImportError:
    print("Warning: py_olamaps not available, using distance calculations only")
    OlaMaps = None
    HAS_OLA_MAPS = False


class TripTimeRecalculator:
    """
    Recalculates time slots for trip data based on place arrangements and travel times.
    """
    
    def __init__(self, ola_api_key: str = "hlN2QjeqJ9hBh2tq1AVvw6O50ilZX4gCfPtbgx6j"):
        """
        Initialize the recalculator with optional OlaMaps API key.
        
        Args:
            ola_api_key: API key for OlaMaps service (optional)
        """
        # Initialize OlaMaps client if available
        if HAS_OLA_MAPS and OlaMaps:
            try:
                self.ola_client = OlaMaps(
                    api_key=ola_api_key,
                    client_id="41b89c52-b013-41a3-b2a4-f0ca47d68d8b",
                    client_secret="IvSD5AfH21DFYh0qHgoqRmLNZIFtyXRM"
                )
                self.has_ola_client = True
            except Exception as e:
                print(f"Failed to initialize OlaMaps client: {e}")
                self.ola_client = None
                self.has_ola_client = False
        else:
            self.ola_client = None
            self.has_ola_client = False
        
        # Default time settings
        self.start_time = "09:00"
        self.lunch_time = "13:00"
        self.dinner_time = "20:00"
        self.night_time = "21:30"
        self.lunch_duration = 90  # minutes
        self.dinner_duration = 90  # minutes
        self.night_duration = 90  # minutes
        self.default_visit_duration = 90  # minutes for attractions

    def haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Calculate the great circle distance between two points on earth in kilometers.
        
        Args:
            lat1, lng1: Latitude and longitude of first point
            lat2, lng2: Latitude and longitude of second point
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Radius of Earth in kilometers
        
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        return R * c

    def calculate_travel_time(self, distance_km: float, mode: str = "driving") -> int:
        """
        Calculate travel time in minutes based on distance and mode of transport.
        
        Args:
            distance_km: Distance in kilometers
            mode: Mode of transport ("driving", "walking")
            
        Returns:
            Travel time in minutes
        """
        if mode == "driving":
            # Average speed in city: 25 km/h
            speed_kmh = 25
        else:  # walking
            speed_kmh = 5
            
        time_hours = distance_km / speed_kmh
        return max(1, int(time_hours * 60))  # Minimum 1 minute

    def time_to_minutes(self, time_str: str) -> int:
        """Convert time string (HH:MM) to minutes since midnight."""
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes

    def minutes_to_time(self, minutes: int) -> str:
        """Convert minutes since midnight to time string (HH:MM)."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def format_duration(self, minutes: int) -> str:
        """Format duration in minutes to HH:MM string."""
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"

    def extract_place_coordinates(self, day_data: Dict[str, Any]) -> List[tuple]:
        """
        Extract coordinates for all places in a day from the new continuous structure.
        
        Args:
            day_data: Single day data from trip JSON
            
        Returns:
            List of (lat, lng, place_key) tuples
        """
        coordinates = []
        
        # Extract all places from the continuous structure
        for key in day_data.get("lat", {}):
            lat = day_data["lat"][key]
            lng = day_data["lng"][key]
            coordinates.append((lat, lng, key))
        
        return coordinates

    def get_place_type(self, day_data: Dict[str, Any], place_key: str) -> str:
        """
        Get the type of place from the data structure.
        
        Args:
            day_data: Single day data from trip JSON
            place_key: Key of the place
            
        Returns:
            Type of place (attraction, restaurant, bar, etc.)
        """
        if "type" in day_data and place_key in day_data["type"]:
            place_type = day_data["type"][place_key]
            
            # Map types to categories
            if place_type in ["restaurant", "vegetarian_restaurant", "meal_takeaway", "food"]:
                return "restaurant"
            elif place_type in ["bar", "night_club", "liquor_store"]:
                return "bar"
            else:
                return "attraction"
        
        # Fallback to checking existing Slots Type if available
        if "Slots" in day_data and "Type" in day_data["Slots"] and place_key in day_data["Slots"]["Type"]:
            slot_type = day_data["Slots"]["Type"][place_key]
            if "Lunch" in slot_type or "Dinner" in slot_type:
                return "restaurant"
            elif "Night" in slot_type:
                return "bar"
            else:
                return "attraction"
        
        return "attraction"  # Default fallback

    def determine_meal_timing(self, day_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Determine which places should be lunch, dinner, and night activities based on type and optimal timing.
        Smart assignment that considers the sequence of places.
        
        Args:
            day_data: Single day data from trip JSON
            
        Returns:
            Dictionary mapping place keys to meal types ("lunch", "dinner", "night", "attraction")
        """
        meal_assignments = {}
        restaurants = []
        bars = []
        attractions = []
        
        # Categorize places by type
        for key in day_data.get("lat", {}):
            place_type = self.get_place_type(day_data, key)
            if place_type == "restaurant":
                restaurants.append(key)
            elif place_type == "bar":
                bars.append(key)
            else:
                attractions.append(key)
        
        # Sort by sequence (assuming keys are numeric strings)
        restaurants.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        bars.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        attractions.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        # Smart meal assignment based on position and count
        total_places = len(restaurants) + len(bars) + len(attractions)
        
        # Assign meal timings intelligently
        lunch_assigned = False
        dinner_assigned = False
        night_assigned = False
        
        # If we have restaurants, assign them based on their position in the sequence
        if restaurants:
            # Find the best restaurant for lunch (should be around middle of day)
            lunch_position = total_places // 3  # Around 1/3 through the day
            dinner_position = (total_places * 2) // 3  # Around 2/3 through the day
            
            # Sort all places to find optimal timing
            all_places = [(int(k) if k.isdigit() else float('inf'), k) for k in day_data.get("lat", {})]
            all_places.sort()
            
            # Find restaurants that are in good positions for meals
            for idx, (_, place_key) in enumerate(all_places):
                if place_key in restaurants:
                    if not lunch_assigned and idx >= lunch_position - 1:
                        meal_assignments[place_key] = "lunch"
                        lunch_assigned = True
                    elif not dinner_assigned and idx >= dinner_position - 1:
                        meal_assignments[place_key] = "dinner"
                        dinner_assigned = True
                    else:
                        meal_assignments[place_key] = "attraction"  # Treat as regular attraction
                        
            # If we still haven't assigned meals, use first available restaurants
            if not lunch_assigned and restaurants:
                for restaurant in restaurants:
                    if restaurant not in meal_assignments:
                        meal_assignments[restaurant] = "lunch"
                        lunch_assigned = True
                        break
                        
            if not dinner_assigned and restaurants:
                for restaurant in restaurants:
                    if restaurant not in meal_assignments and meal_assignments.get(restaurant) != "lunch":
                        meal_assignments[restaurant] = "dinner"
                        dinner_assigned = True
                        break
        
        # Assign night activities (bars) - prefer later in sequence
        if bars and not night_assigned:
            # Use the last bar in sequence for night activity
            meal_assignments[bars[-1]] = "night"
            night_assigned = True
            # Other bars become attractions
            for bar in bars[:-1]:
                meal_assignments[bar] = "attraction"
        
        # All other places are attractions
        for attraction in attractions:
            meal_assignments[attraction] = "attraction"
            
        # Fill in any remaining places not assigned
        for key in day_data.get("lat", {}):
            if key not in meal_assignments:
                meal_assignments[key] = "attraction"
        
        return meal_assignments
    async def recalculate_day_slots(self, day_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recalculate time slots for a single day based on place arrangements using the new continuous structure.
        Calculate optimal slot durations based on available time and travel requirements.
        
        Args:
            day_data: Single day data from trip JSON
            
        Returns:
            Updated day data with recalculated slots
        """
        # Extract all coordinates
        coordinates = self.extract_place_coordinates(day_data)
        coord_dict = {key: (lat, lng) for lat, lng, key in coordinates}
        
        # Determine meal assignments
        meal_assignments = self.determine_meal_timing(day_data)
        
        # Get all place keys in sequence
        place_keys = list(day_data.get("lat", {}).keys())
        place_keys.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        # Calculate all travel times first
        travel_times = {}
        total_travel_time = 0
        
        for i, place_key in enumerate(place_keys):
            if i == 0:
                travel_times[place_key] = 0
            else:
                prev_key = place_keys[i-1]
                if prev_key in coord_dict and place_key in coord_dict:
                    prev_coord = coord_dict[prev_key]
                    curr_coord = coord_dict[place_key]
                    travel_duration = await self.calculate_travel_duration(
                        prev_coord[0], prev_coord[1], curr_coord[0], curr_coord[1]
                    )
                    travel_times[place_key] = travel_duration
                    total_travel_time += travel_duration
                else:
                    travel_times[place_key] = 15  # Default 15 minutes
                    total_travel_time += 15
        
        # Calculate available time and optimal slot durations
        start_time_mins = self.time_to_minutes(self.start_time)  # 09:00
        end_time_mins = self.time_to_minutes("22:00")  # End of regular day
        total_available_time = end_time_mins - start_time_mins  # 13 hours = 780 minutes
        
        # Separate places by type for duration calculation
        attractions = []
        lunch_place = None
        dinner_place = None
        night_place = None
        
        for place_key in place_keys:
            meal_type = meal_assignments.get(place_key, "attraction")
            if meal_type == "lunch":
                lunch_place = place_key
            elif meal_type == "dinner":
                dinner_place = place_key
            elif meal_type == "night":
                night_place = place_key
            else:
                attractions.append(place_key)
        
        # Reserve fixed time for meals
        reserved_meal_time = 0
        if lunch_place:
            reserved_meal_time += self.lunch_duration  # 90 minutes
        if dinner_place:
            reserved_meal_time += self.dinner_duration  # 90 minutes
        if night_place:
            reserved_meal_time += self.night_duration  # 90 minutes
        
        # Calculate time available for attractions
        available_for_attractions = total_available_time - total_travel_time - reserved_meal_time
        
        # Calculate optimal duration per attraction
        num_attractions = len(attractions)
        if num_attractions > 0:
            attraction_duration = max(45, min(120, available_for_attractions // num_attractions))  # Between 45-120 minutes
        else:
            attraction_duration = 90  # Default
        
        # Initialize slots
        new_slots = {
            "Start": {},
            "End": {},
            "Slot Duration": {},
            "Travel Duration": {},
            "Type": {}
        }
        
        # Calculate actual slots with proper time management
        current_time = start_time_mins
        lunch_time_mins = self.time_to_minutes(self.lunch_time)  # 13:00 = 780 minutes
        dinner_time_mins = self.time_to_minutes(self.dinner_time)  # 20:00 = 1200 minutes
        night_time_mins = self.time_to_minutes(self.night_time)  # 21:30 = 1290 minutes
        day_end_mins = self.time_to_minutes("23:59")  # End of day = 1439 minutes
        
        # Process places in sequence with proper time constraints
        lunch_scheduled = False
        dinner_scheduled = False
        night_scheduled = False
        
        for i, place_key in enumerate(place_keys):
            # Add travel time (but ensure we don't exceed day boundaries)
            travel_time = travel_times[place_key]
            current_time += travel_time
            
            # Check if we're running out of time for the day
            if current_time >= day_end_mins:
                print(f"Warning: Day {i+1} schedule exceeds 24 hours, truncating remaining places")
                break
            
            # Determine visit duration and type based on meal assignment - no strict timing windows
            meal_type = meal_assignments.get(place_key, "attraction")
            
            if meal_type == "lunch" and not lunch_scheduled:
                # Schedule lunch naturally in the sequence
                visit_duration = self.lunch_duration
                slot_type = "Lunch"
                lunch_scheduled = True
                    
            elif meal_type == "dinner" and not dinner_scheduled:
                # Schedule dinner naturally in the sequence
                visit_duration = self.dinner_duration
                slot_type = "Dinner"
                dinner_scheduled = True
                    
            elif meal_type == "night" and not night_scheduled:
                # Schedule night activity naturally in the sequence
                visit_duration = self.night_duration
                slot_type = "Night Activity"
                night_scheduled = True
                
            else:
                # Regular attraction
                visit_duration = attraction_duration
                slot_type = "Attraction"
                
                # Adjust duration if we're running out of time
                remaining_places = len([p for j, p in enumerate(place_keys) if j > i])
                time_left = day_end_mins - current_time
                
                if remaining_places > 1:
                    # Reserve time for remaining places
                    estimated_remaining_travel = sum(travel_times.get(place_keys[j], 15) for j in range(i+1, len(place_keys)))
                    available_for_visits = time_left - estimated_remaining_travel
                    max_duration = max(30, available_for_visits // remaining_places)
                    visit_duration = min(visit_duration, max_duration)
                else:
                    # Last place - use remaining time but cap at reasonable maximum
                    visit_duration = min(visit_duration, time_left - 30)  # Leave 30 min buffer
                
                # Ensure minimum visit time
                visit_duration = max(30, visit_duration)
            
            start_time = current_time
            end_time = current_time + visit_duration
            
            # Final check to ensure we don't exceed day boundary
            if end_time > day_end_mins:
                end_time = day_end_mins
                visit_duration = end_time - start_time
                if visit_duration < 30:  # If less than 30 minutes, skip this place
                    break
            
            # Store slot information
            new_slots["Start"][place_key] = self.minutes_to_time(start_time)
            new_slots["End"][place_key] = self.minutes_to_time(end_time)
            new_slots["Slot Duration"][place_key] = self.format_duration(visit_duration)
            new_slots["Travel Duration"][place_key] = travel_time
            new_slots["Type"][place_key] = slot_type
            
            current_time = end_time

        # Update the day data
        updated_day_data = day_data.copy()
        updated_day_data["Slots"] = new_slots
        
        return updated_day_data

    async def recalculate_trip_times(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recalculate time slots for entire trip data.
        
        Args:
            trip_data: Complete trip data JSON (can have "trip" wrapper or be direct day data)
            
        Returns:
            Updated trip data with recalculated time slots
        """
        updated_trip_data = trip_data.copy()
        
        # Check if trip_data has the old "trip" wrapper structure
        if "trip" in trip_data:
            # Process each day within trip wrapper
            for day_key, day_data in trip_data["trip"].items():
                if day_key.startswith("Day "):
                    print(f"Recalculating times for {day_key}...")
                    updated_day_data = await self.recalculate_day_slots(day_data)
                    updated_trip_data["trip"][day_key] = updated_day_data
        else:
            # Process direct day data structure (new format)
            for day_key, day_data in trip_data.items():
                if day_key.startswith("Day "):
                    print(f"Recalculating times for {day_key}...")
                    updated_day_data = await self.recalculate_day_slots(day_data)
                    updated_trip_data[day_key] = updated_day_data
        
        return updated_trip_data

    async def get_distance_matrix(self, start_lat, start_lng, end_lat, end_lng, max_retries=3):
        """Get distance matrix between two points with retry logic"""
        # If OlaMaps client is not available, return default values
        if not self.has_ola_client:
            distance_km = self.haversine_distance(start_lat, start_lng, end_lat, end_lng)
            return {
                'rows': [{
                    'elements': [{
                        'distance': {'value': int(distance_km * 1000), 'text': f'{distance_km:.1f} km'},
                        'duration': max(5, int(distance_km * 240)),  # ~25 km/h average speed in seconds
                        'status': 'OK'
                    }]
                }]
            }
        
        start_point = f"{start_lat},{start_lng}"
        end_point = f"{end_lat},{end_lng}"
        
        retry_delays = [1, 2, 4]
        
        for attempt in range(max_retries):
            try:
                # Validate coordinates
                if not all(isinstance(x, (int, float)) for x in [start_lat, start_lng, end_lat, end_lng]):
                    raise ValueError("Invalid coordinate values")
                
                if not (-90 <= start_lat <= 90 and -90 <= end_lat <= 90 and
                       -180 <= start_lng <= 180 and -180 <= end_lng <= 180):
                    raise ValueError("Coordinates out of valid range")
                
                distance_matrix = self.ola_client.routing.distance_matrix(start_point, end_point)
                
                if not distance_matrix or not isinstance(distance_matrix, dict):
                    raise ValueError("Invalid distance matrix response")
                
                if 'rows' not in distance_matrix or not distance_matrix['rows']:
                    raise ValueError("No distance data in response")
                
                first_row = distance_matrix['rows'][0]
                if 'elements' not in first_row or not first_row['elements']:
                    raise ValueError("No route elements in response")
                
                first_element = first_row['elements'][0]
                if 'status' not in first_element or first_element['status'] != 'OK':
                    if first_element['status'] == 'NOT_FOUND':
                        # Calculate fallback distance using Haversine formula
                        R = 6371
                        lat1, lon1 = radians(start_lat), radians(start_lng)
                        lat2, lon2 = radians(end_lat), radians(end_lng)
                        
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * atan2(sqrt(a), sqrt(1-a))
                        distance = R * c
                        
                        return {
                            'rows': [{
                                'elements': [{
                                    'distance': {'value': int(distance * 1000), 'text': f"{distance:.1f} km"},
                                    'duration': int(distance * 200),
                                    'status': 'FALLBACK'
                                }]
                            }]
                        }
                    
                    raise ValueError(f"Route status: {first_element['status']}")
                
                return distance_matrix
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    await asyncio.sleep(retry_delays[attempt])
                    continue
                else:
                    print(f"All attempts failed: {str(e)}")
                    return {
                        'rows': [{
                            'elements': [{
                                'distance': {'value': 0, 'text': 'Unknown'},
                                'duration': 0,
                                'status': 'ERROR',
                                'error': str(e)
                            }]
                        }]
                    }

    async def calculate_travel_duration(self, start_lat, start_lng, end_lat, end_lng):
        """Calculate travel duration between two points in minutes with improved error handling"""
        try:
            distance_matrix = await self.get_distance_matrix(start_lat, start_lng, end_lat, end_lng)
            
            if 'rows' in distance_matrix and distance_matrix['rows']:
                element = distance_matrix['rows'][0]['elements'][0]
                
                if element.get('status') == 'OK':
                    duration = element.get('duration')
                    if isinstance(duration, dict) and 'value' in duration:
                        duration_seconds = duration['value']
                    elif isinstance(duration, (int, float)):
                        duration_seconds = duration
                    else:
                        print(f"Invalid duration format: {duration}")
                        # Fallback calculation
                        distance_km = self.haversine_distance(start_lat, start_lng, end_lat, end_lng)
                        duration_seconds = max(300, int(distance_km * 144))  # 25 km/h average
                    
                    duration_minutes = max(5, duration_seconds // 60)  # Minimum 5 minutes
                    return int(duration_minutes)
                else:
                    print(f"Distance matrix status not OK: {element.get('status')}")
                    # Fallback calculation
                    distance_km = self.haversine_distance(start_lat, start_lng, end_lat, end_lng)
                    duration_minutes = max(5, int(distance_km * 2.4))  # Approx 25 km/h
                    return int(duration_minutes)
            else:
                print(f"Invalid distance matrix response: {distance_matrix}")
                # Fallback calculation
                distance_km = self.haversine_distance(start_lat, start_lng, end_lat, end_lng)
                duration_minutes = max(5, int(distance_km * 2.4))  # Approx 25 km/h
                return int(duration_minutes)
                
        except Exception as e:
            print(f"Error calculating travel duration: {e}")
            # Ultimate fallback - use Haversine distance
            distance_km = self.haversine_distance(start_lat, start_lng, end_lat, end_lng)
            duration_minutes = max(5, int(distance_km * 2.4))  # Approx 25 km/h
            return int(duration_minutes)

    def convert_minutes(self, minutes):
        """Convert minutes to HH:MM format"""
        hours = minutes // 60
        remaining_minutes = minutes % 60
        return f"{int(hours):02}:{int(remaining_minutes):02}"

    async def calculate_durations_for_rearranged_trip(self, trip_data, travel_schedule=None, place_types=None):
        """Calculate travel durations for rearranged trip data with new continuous structure"""
        durations = []
        
        for day_key in trip_data:
            if not day_key.startswith('Day '):
                continue
                
            day_data = trip_data[day_key]
            duration = []
            
            # Get coordinates for all places in sequence
            places_coords = []
            place_keys = list(day_data.get('lat', {}).keys())
            place_keys.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
            
            for place_key in place_keys:
                lat = day_data['lat'][place_key]
                lng = day_data['lng'][place_key]
                places_coords.append((lat, lng))
            
            # Calculate travel durations between consecutive places
            for i in range(len(places_coords) - 1):
                start_lat, start_lng = places_coords[i]
                end_lat, end_lng = places_coords[i + 1]
                travel_duration = await self.calculate_travel_duration(start_lat, start_lng, end_lat, end_lng)
                duration.append(travel_duration)
            
            # Add final duration (return to hotel or end point)
            if places_coords:
                duration.append(15)  # Default return time
            
            durations.append(duration)
        
        return durations

    def recalculate_slots_with_new_durations(self, trip_data, travel_durations, travel_schedule=None, place_types=None):
        """Recalculate time slots based on new travel durations for continuous structure with proper duration calculation"""
        
        # Default schedule configuration if not provided
        if not travel_schedule:
            travel_schedule = "Explorer"
            
        schedule_config = {
            "Explorer": {"hours_per_day": 12, "min_attraction_duration": 60, "max_attraction_duration": 120},
            "Adventurer": {"hours_per_day": 14, "min_attraction_duration": 45, "max_attraction_duration": 90},
            "Leisure": {"hours_per_day": 10, "min_attraction_duration": 90, "max_attraction_duration": 150}
        }
        
        config = schedule_config.get(travel_schedule, schedule_config["Explorer"])
        daily_hours = config["hours_per_day"]
        min_attraction_duration = config["min_attraction_duration"]
        max_attraction_duration = config["max_attraction_duration"]
        
        # Fixed meal durations
        meal_durations = {
            "lunch": 90,
            "dinner": 90,
            "night": 90
        }
        
        # Default times
        start_time = datetime.strptime("09:00", "%H:%M")
        lunch_time = datetime.strptime("13:00", "%H:%M")
        dinner_time = datetime.strptime("20:00", "%H:%M")
        night_time = datetime.strptime("22:00", "%H:%M")
        
        day_index = 0
        
        for day_key in trip_data:
            if not day_key.startswith('Day '):
                continue
                
            day_data = trip_data[day_key]
            
            if day_index >= len(travel_durations):
                continue
                
            durations = travel_durations[day_index]
            meal_assignments = self.determine_meal_timing(day_data)
            
            # Get all place keys in sequence
            place_keys = list(day_data.get('lat', {}).keys())
            place_keys.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
            
            # Calculate total available time
            total_available_minutes = daily_hours * 60  # Convert hours to minutes
            total_travel_time = sum(durations[:len(place_keys)])
            
            # Calculate time needed for meals
            meal_time = 0
            attraction_count = 0
            
            for place_key in place_keys:
                meal_type = meal_assignments.get(place_key, "attraction")
                if meal_type in meal_durations:
                    meal_time += meal_durations[meal_type]
                else:
                    attraction_count += 1
            
            # Calculate time available for attractions
            available_for_attractions = total_available_minutes - total_travel_time - meal_time
            
            # Calculate optimal duration per attraction
            if attraction_count > 0:
                base_attraction_duration = available_for_attractions // attraction_count
                # Clamp to reasonable limits
                attraction_duration = max(min_attraction_duration, 
                                        min(max_attraction_duration, base_attraction_duration))
            else:
                attraction_duration = min_attraction_duration
            
            # Build new slots
            slots = {}
            current_time = start_time
            
            for i, place_key in enumerate(place_keys):
                # Get travel time to this place
                travel_time = durations[i] if i < len(durations) else 15
                
                # Add travel time
                current_time += timedelta(minutes=travel_time)
                
                # Determine visit duration and type based on meal assignment
                meal_type = meal_assignments.get(place_key, "attraction")
                
                if meal_type == "lunch":
                    # Ensure lunch is around lunch time
                    if current_time < lunch_time:
                        current_time = lunch_time
                    visit_duration = meal_durations["lunch"]
                    slot_type = "Lunch"
                elif meal_type == "dinner":
                    # Ensure dinner is around dinner time
                    if current_time < dinner_time:
                        current_time = dinner_time
                    visit_duration = meal_durations["dinner"]
                    slot_type = "Dinner"
                elif meal_type == "night":
                    # Ensure night activity is after night time
                    if current_time < night_time:
                        current_time = night_time
                    visit_duration = meal_durations["night"]
                    slot_type = "Night Activity"
                else:
                    # Regular attraction - use calculated optimal duration
                    visit_duration = attraction_duration
                    slot_type = "Attraction"
                    
                    # Dynamic adjustment based on remaining time
                    remaining_places = len(place_keys) - i - 1
                    if remaining_places > 0:
                        # Calculate remaining travel time
                        remaining_travel = sum(durations[j] for j in range(i+1, min(len(durations), len(place_keys))))
                        
                        # Calculate time until end of day (22:00)
                        end_of_day = datetime.strptime("22:00", "%H:%M")
                        if current_time.time() > end_of_day.time():
                            end_of_day += timedelta(days=1)  # Next day
                        
                        remaining_time = (end_of_day - current_time).total_seconds() / 60
                        
                        # Adjust duration if running late
                        if remaining_time > 0:
                            max_duration = (remaining_time - remaining_travel) / (remaining_places + 1)
                            if max_duration > 0:
                                visit_duration = min(visit_duration, int(max_duration))
                    
                    # Ensure minimum duration
                    visit_duration = max(30, visit_duration)
                
                slot_start = current_time
                slot_end = current_time + timedelta(minutes=visit_duration)
                
                slots[place_key] = {
                    "Start": slot_start.strftime("%H:%M"),
                    "End": slot_end.strftime("%H:%M"),
                    "Slot Duration": self.convert_minutes(visit_duration),
                    "Travel Duration": travel_time,
                    "Type": slot_type
                }
                
                current_time = slot_end
            
            # Update the trip data with new slots
            try:
                # Convert to the expected format
                updated_slots = {
                    "Start": {},
                    "End": {},
                    "Slot Duration": {},
                    "Travel Duration": {},
                    "Type": {}
                }
                
                for place_key, slot_info in slots.items():
                    updated_slots["Start"][place_key] = slot_info["Start"]
                    updated_slots["End"][place_key] = slot_info["End"]
                    updated_slots["Slot Duration"][place_key] = slot_info["Slot Duration"]
                    updated_slots["Travel Duration"][place_key] = slot_info["Travel Duration"]
                    updated_slots["Type"][place_key] = slot_info["Type"]
                
                trip_data[day_key]["Slots"] = updated_slots
                
            except Exception as e:
                print(f"Error creating slots for {day_key}: {e}")
                trip_data[day_key]["Slots"] = {"Error": f"Slot creation failed: {str(e)}"}
            
            day_index += 1
        
        return trip_data

    async def recalculate_trip_timing(self, trip_data, travel_schedule=None, place_types=None):
        """Main function to recalculate trip timing for rearranged data with proper duration calculation"""
        try:
            # Always use the improved recalculate_trip_times method that has proper time boundary handling
            print("Using improved recalculation method with proper time boundaries...")
            updated_trip = await self.recalculate_trip_times(trip_data)
            return updated_trip
            
        except Exception as e:
            print(f"Error recalculating trip timing: {e}")
            import traceback
            traceback.print_exc()
            return trip_data  # Return original data if calculation fails


# Convenience function for direct use
async def recalculate_trip_times(trip_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to recalculate trip times using the new structure.
    
    Args:
        trip_data: Complete trip data JSON (supports both old and new formats)
        
    Returns:
        Updated trip data with recalculated time slots
    """
    recalculator = TripTimeRecalculator()
    return await recalculator.recalculate_trip_times(trip_data)


# Example usage for testing
if __name__ == "__main__":
    # Example trip data matching the new continuous structure
    sample_trip_data = {
        "Day 1": {
            "Name": {
                "0": "Emu enclosure",
                "1": "Ghadiyal enclosure", 
                "2": "DHABALOGY",
                "3": "श्री चारभुजा गौशाला वरदा",
                "4": "Crocodile enclosure",
                "5": "Fox enclosure",
                "6": "Bawarchi Restaurant",
                "7": "Rootage Restaurant And Lounge"
            },
            "lat": {
                "0": 24.593040799999997,
                "1": 24.591763399999998,
                "2": 24.7356166,
                "3": 24.655583500000002,
                "4": 24.5917475,
                "5": 24.589900099999998,
                "6": 24.5849393,
                "7": 24.5805783
            },
            "lng": {
                "0": 73.6500585,
                "1": 73.64790769999999,
                "2": 73.7409374,
                "3": 73.6079237,
                "4": 73.6485538,
                "5": 73.64585749999999,
                "6": 73.6952694,
                "7": 73.6968648
            },
            "type": {
                "0": "zoo",
                "1": "zoo",
                "2": "vegetarian_restaurant",
                "3": "zoo",
                "4": "zoo",
                "5": "zoo",
                "6": "vegetarian_restaurant",
                "7": "bar"
            },
            "Weekday": "Wednesday"
        }
    }
    
    async def test_recalculation():
        recalculator = TripTimeRecalculator()
        updated_data = await recalculator.recalculate_trip_times(sample_trip_data)
        print("Updated trip data slots:")
        if "Day 1" in updated_data and "Slots" in updated_data["Day 1"]:
            print(json.dumps(updated_data["Day 1"]["Slots"], indent=2))
        else:
            print("No slots found in updated data")
            print(json.dumps(updated_data, indent=2))
    
    # Run the test
    # asyncio.run(test_recalculation())