# Trip Modification with Time Recalculation

This document explains how to use the `/trips/modify` endpoint with automatic time recalculation functionality.

## Overview

The `/trips/modify` endpoint allows users to submit rearranged trip data and automatically recalculates optimal time slots based on:
- Travel distances between places
- Travel schedule preference (Leisure, Explorer, Adventurer)
- Place types and meal arrangements
- Realistic travel times

## Endpoint Details

**URL:** `POST /trips/modify`
**Authentication:** Bearer token required
**Content-Type:** `application/json`

## Request Body Structure

The request body should match the `TripGenerationData` model:

```json
{
  "city_name": "City, State, Country",
  "place_types": ["museum", "tourist_attraction", "hindu_temple"],
  "no_of_days": 1,
  "travel_schedule": "Explorer",
  "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "trip": {
    "Day 1": {
      "Name": {
        "0": "Place 1 Name",
        "1": "Place 2 Name"
      },
      "lat": {
        "0": 24.5953732,
        "1": 24.5931853
      },
      "lng": {
        "0": 73.6872296,
        "1": 73.6396056
      },
      "Weekday": "Tuesday",
      "Lunch": {
        "Name": {"0": "Restaurant Name"},
        "lat": {"0": 24.5849393},
        "lng": {"0": 73.6952694}
      },
      "Dinner": {
        "Name": {"0": "Restaurant Name"},
        "lat": {"0": 24.5723618},
        "lng": {"0": 73.6996646}
      },
      "Night": {
        "Name": {"0": "Night Venue"},
        "lat": {"0": 24.5805783},
        "lng": {"0": 73.6968648}
      }
    }
  }
}
```

## Travel Schedule Options

### Leisure
- **Slots:** 3 slots per day
- **Duration:** 45+ minutes per slot
- **Pace:** Relaxed with longer stays
- **Best for:** Couples, families, relaxed travelers

### Explorer
- **Slots:** 4 slots per day
- **Duration:** 30+ minutes per slot
- **Pace:** Moderate exploration
- **Best for:** Most travelers, balanced itinerary

### Adventurer
- **Slots:** 5 slots per day
- **Duration:** 25+ minutes per slot
- **Pace:** Fast-paced with many locations
- **Best for:** Active travelers, short trips

## Response Format

```json
{
  "success": true,
  "message": "Trip modified and times recalculated successfully",
  "trip_data": {
    "city_name": "Udaipur, Rajasthan, India",
    "place_types": ["museum", "tourist_attraction"],
    "travel_schedule": "Explorer",
    "trip": {
      "Day 1": {
        "Name": { "0": "Place 1", "1": "Place 2" },
        "lat": { "0": 24.595, "1": 24.593 },
        "lng": { "0": 73.687, "1": 73.640 },
        "Slots": {
          "Start": {
            "0": "09:00",
            "1": "11:09",
            "Lunch": "13:00",
            "Dinner": "20:00"
          },
          "End": {
            "0": "10:51",
            "1": "13:00",
            "Lunch": "14:30",
            "Dinner": "21:30"
          },
          "Slot Duration": {
            "0": "01:51",
            "1": "01:51",
            "Lunch": "01:30",
            "Dinner": "01:30"
          },
          "Travel Duration": {
            "0": 18,
            "1": 15,
            "Lunch": 15,
            "Dinner": 18
          },
          "Type": {
            "0": "Morning Activity",
            "1": "Morning Activity",
            "Lunch": "Meal",
            "Dinner": "Meal"
          }
        }
      }
    }
  },
  "user": "user@example.com",
  "recalculation_info": {
    "travel_schedule": "Explorer",
    "place_types": ["museum", "tourist_attraction"],
    "days_processed": 1
  }
}
```

## Time Calculation Features

### Automatic Calculations
- **Travel Time:** Based on real coordinates using Haversine distance
- **Visit Duration:** Optimized based on travel schedule
- **Meal Timing:** Fixed lunch (13:00-14:30), dinner (20:00-21:30)
- **Slot Optimization:** Balances travel time with visit duration

### Smart Scheduling
- **Morning Slots:** Start at 09:00
- **Lunch Break:** Automatically inserted at 13:00
- **Evening Slots:** Optimized around dinner time
- **Night Activities:** Added for eligible place types (bars, clubs)

### Error Handling
- **Invalid Coordinates:** Falls back to estimated travel times
- **Missing Data:** Uses sensible defaults
- **API Failures:** Graceful degradation with Haversine calculations

## Usage Examples

### Basic Usage (Explorer Schedule)
```bash
curl -X POST "http://localhost:8000/trips/modify" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d @trip_data.json
```

### Testing with HTTP Client
Use the provided `test_trip_modify.http` file with VS Code REST Client extension or similar tools.

### Programmatic Usage
```python
import requests

headers = {
    "Authorization": "Bearer YOUR_TOKEN",
    "Content-Type": "application/json"
}

trip_data = {
    # Your trip data here
}

response = requests.post(
    "http://localhost:8000/trips/modify",
    headers=headers,
    json=trip_data
)

updated_trip = response.json()
```

## Error Responses

### 400 Bad Request
- Missing required trip data
- Invalid coordinates
- Malformed request body

### 401 Unauthorized
- Missing or invalid authentication token

### 500 Internal Server Error
- API communication failures
- Unexpected calculation errors
- Server-side processing issues

## Best Practices

1. **Coordinate Accuracy:** Ensure lat/lng values are accurate for better travel time calculations
2. **Place Order:** Arrange places in your preferred visit order in the request
3. **Meal Selection:** Choose restaurants close to your attractions for optimal routing
4. **Schedule Choice:** Select travel schedule based on your pace preference
5. **Error Handling:** Always check the response status and handle errors gracefully

## Limitations

- Maximum 10 places per day for optimal performance
- Coordinates must be valid lat/lng values
- Night activities only added for compatible place types
- Travel times calculated for road travel (driving/walking)

## Development Notes

The time recalculation engine uses:
- **OlaMaps API** for precise travel time calculations
- **Haversine formula** as fallback for distance estimation
- **Pandas** for efficient slot data manipulation
- **Async processing** for better performance

For technical details, see `trip_time_recalculator.py` module.
