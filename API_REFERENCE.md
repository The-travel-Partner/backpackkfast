# Backpackk API Reference

Quick reference guide for all API endpoints in the Backpackk travel platform.

## Base URL
- **Production**: `https://backpackkfast-fcvonqkgya-el.a.run.app`
- **Alternative**: `https://backpackk-cloud.el.r.appspot.com`
- **Local**: `http://localhost:8000`

---

## 🔐 Authentication Endpoints

### Register New User
```http
POST /register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "securepassword",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response:**
```json
{
  "message": "User registered successfully. Please check your email to verify your account."
}
```

### Login
```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=user@example.com&password=securepassword
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer",
  "first_name": "John"
}
```

### Verify Email
```http
GET /verify/{user_id}/{token}
```

### Get Current User
```http
GET /users/me
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "disabled": false,
  "verified": true
}
```

### Google OAuth Login
```http
GET /google
```
Redirects to Google authentication.

### Forgot Password
```http
POST /forgot-password
Content-Type: application/json

{
  "email": "user@example.com"
}
```

### Reset Password
```http
POST /reset-password
Content-Type: application/json

{
  "token": "reset_token_from_email",
  "new_password": "newsecurepassword"
}
```

---

## 🗺️ Trip Generation Endpoints

### Generate New Trip
```http
POST /tripgenerator
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "city_name": "Udaipur, Rajasthan, India",
  "place_types": ["museum", "tourist_attraction", "hindu_temple"],
  "travel_schedule": "Explorer",
  "no_of_days": 2,
  "weekdays": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
}
```

**Travel Schedules:**
- `Leisure` - 3 places/day, relaxed pace
- `Explorer` - 4 places/day, moderate pace
- `Adventurer` - 5 places/day, fast pace

**Response:** Complete trip itinerary with continuous structure.

### Calculate Available Days
```http
POST /number/of/available/days
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "city_name": "Udaipur, Rajasthan, India",
  "place_types": ["museum", "tourist_attraction"],
  "travel_schedule": "Explorer",
  "weekdays": ["Monday", "Tuesday", "Wednesday"]
}
```

**Response:**
```json
{
  "Number": 3
}
```

### Modify Trip with Time Recalculation
```http
POST /trips/modify
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "city_name": "Udaipur, Rajasthan, India",
  "place_types": ["museum", "tourist_attraction"],
  "no_of_days": 1,
  "travel_schedule": "Explorer",
  "trip": {
    "Day 1": {
      "Name": {"0": "Place A", "1": "Place B"},
      "lat": {"0": 24.595, "1": 24.593},
      "lng": {"0": 73.687, "1": 73.640},
      "Weekday": "Monday"
    }
  }
}
```

See [TRIP_MODIFICATION_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/TRIP_MODIFICATION_README.md) for detailed documentation.

### Get User Trips
```http
GET /gettrips?all=false
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `all` (boolean): `true` for all trips, `false` for latest 3 (default: `false`)

---

## 📍 Places & Discovery Endpoints

### Get Places for City
```http
POST /getplaces
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "city_name": "Udaipur, Rajasthan, India",
  "place_types": ["museum", "tourist_attraction", "restaurant"]
}
```

**Response:** Array of places with ratings, coordinates, opening hours, etc.

### Get All Places (Alternative)
```http
GET /getplaces/all?cityname=Udaipur, Rajasthan, India
Authorization: Bearer {access_token}
```

### City Name Autocomplete
```http
GET /autocomplete?query=Udai
```

**Response:**
```json
{
  "cities": [
    "Udaipur, Rajasthan, India",
    "Udaipur, Tripura, India"
  ]
}
```

### Get Place Details
```http
POST /place
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "place_id": "ChIJU11SpUvlZzkRlSL_l4aYtWw",
  "include_photos": true
}
```

**Response:**
```json
{
  "place_id": "ChIJU11SpUvlZzkRlSL_l4aYtWw",
  "name": "City Palace",
  "location": {"latitude": 24.576, "longitude": 73.683},
  "rating": 4.7,
  "vicinity": "Old City, Udaipur",
  "photos": [
    {
      "image": "base64_encoded_image",
      "content_type": "image/jpeg"
    }
  ],
  "timestamp": "2025-11-26T13:00:00"
}
```

### Find Nearby Places
```http
POST /nearby-places
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "latitude": 24.5761,
  "longitude": 73.6833,
  "placeid": "ChIJU11SpUvlZzkRlSL_l4aYtWw",
  "radius": 3000,
  "max_results": 20,
  "include_photos": true
}
```

**Response:**
```json
{
  "attractions": [
    {
      "place_id": "...",
      "name": "Jagdish Temple",
      "location": {"latitude": 24.580, "longitude": 73.683},
      "rating": 4.5,
      "photos": [...]
    }
  ],
  "restaurants": [
    {
      "place_id": "...",
      "name": "Ambrai Restaurant",
      "location": {"latitude": 24.577, "longitude": 73.681},
      "rating": 4.6,
      "photos": [...]
    }
  ],
  "total_results": 15
}
```

### Get AI Place Description
```http
GET /placedescription?placename=City Palace&cityname=Udaipur, Rajasthan, India
Authorization: Bearer {access_token}
```

**Response:**
```json
{
  "description": "The City Palace in Udaipur is a majestic complex overlooking Lake Pichola, built in 1559. Known for its intricate Rajput and Mughal architecture, it houses museums showcasing royal artifacts, paintings, and crystal galleries."
}
```

---

## 🖼️ Photo Endpoints

### Get Photos by Reference
```http
POST /getphotos
Content-Type: application/json

{
  "photoref": [
    "places/ChIJ.../photos/...",
    "places/ChIJ.../photos/..."
  ]
}
```

### Get Photos by Place ID
```http
POST /getphotos/byplaceid
Content-Type: application/json

{
  "place_id": "ChIJU11SpUvlZzkRlSL_l4aYtWw"
}
```

**Response:**
```json
{
  "images": [
    {
      "image": "base64_encoded_image_data",
      "content_type": "image/jpeg"
    }
  ]
}
```

### Get Single Photo
```http
GET /getphoto?name=places/ChIJ.../photos/AfLeUgN...
Authorization: Bearer {access_token}
```

Returns JPEG image directly.

---

## 🗺️ Maps Endpoints

### Generate Static Map URL
```http
GET /generate-map-blob?placename=City Palace&location=24.576,73.683
Authorization: Bearer {access_token}
```

**Response:** Static Google Maps URL as text stream.

### Get Street View Embed URL
```http
GET /generate-map-blob/streetview?location=24.576,73.683
Authorization: Bearer {access_token}
```

**Response:**
```
https://www.google.com/maps/embed/v1/streetview?location=24.576,73.683&key=...
```

### Get Place View Embed URL
```http
GET /generate-map-blob/view?placeid=ChIJU11SpUvlZzkRlSL_l4aYtWw
Authorization: Bearer {access_token}
```

**Response:**
```
https://www.google.com/maps/embed/v1/place?q=place_id:ChIJU11SpUvlZzkRlSL_l4aYtWw&key=...
```

---

## ✈️ Flights Endpoints

### Search Flights
```http
POST /flights/search
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "departure_id": "DEL",
  "arrival_id": "BOM",
  "outbound_date": "2025-12-01",
  "return_date": "2025-12-10",
  "type": 1,
  "travel_class": 1,
  "adults": 2,
  "children": 0,
  "infants_in_seat": 0,
  "infants_on_lap": 0,
  "currency": "INR",
  "gl": "in",
  "hl": "en",
  "sort_by": 2
}
```

**Parameters:**
- `departure_id` (required): Departure airport code (e.g., "JFK", "DEL")
- `arrival_id` (required): Arrival airport code (e.g., "LAX", "BOM")
- `outbound_date` (required): Departure date in YYYY-MM-DD format
- `return_date` (optional): Return date, required for round trip (type=1)
- `type` (optional): Flight type - 1=Round trip (default), 2=One way, 3=Multi-city
- `travel_class` (optional): 1=Economy (default), 2=Premium economy, 3=Business, 4=First
- `adults` (optional): Number of adults, default=1
- `children` (optional): Number of children, default=0
- `infants_in_seat` (optional): Number of infants in seat, default=0
- `infants_on_lap` (optional): Number of infants on lap, default=0
- `currency` (optional): Currency code, default="USD"
- `gl` (optional): Country code, default="us"
- `hl` (optional): Language code, default="en"
- `sort_by` (optional): Sort order - 1=Top flights, 2=Price, 3=Departure time, 4=Arrival time, 5=Duration, 6=Emissions

**Response:**
```json
{
  "search_parameters": {
    "engine": "google_flights",
    "departure_id": "DEL",
    "arrival_id": "BOM",
    "outbound_date": "2025-12-01",
    "return_date": "2025-12-10"
  },
  "best_flights": [
    {
      "flights": [
        {
          "departure_airport": {
            "name": "Indira Gandhi International Airport",
            "id": "DEL",
            "time": "2025-12-01 06:00"
          },
          "arrival_airport": {
            "name": "Chhatrapati Shivaji Maharaj International Airport",
            "id": "BOM",
            "time": "2025-12-01 08:15"
          },
          "duration": 135,
          "airplane": "Boeing 737",
          "airline": "Air India",
          "airline_logo": "https://...",
          "travel_class": "Economy",
          "flight_number": "AI 860",
          "legroom": "31 in"
        }
      ],
      "total_duration": 135,
      "carbon_emissions": {
        "this_flight": 85000,
        "typical_for_this_route": 90000,
        "difference_percent": -6
      },
      "price": 8500,
      "type": "Round trip",
      "departure_token": "..."
    }
  ],
  "other_flights": [...],
  "price_insights": {
    "lowest_price": 8500,
    "price_level": "typical",
    "typical_price_range": [7500, 12000]
  },
  "airports": [...],
  "user": "user@example.com",
  "search_metadata": {
    "total_results": 15,
    "currency": "INR",
    "search_timestamp": "2025-11-26T07:45:00"
  }
}
```

**⚠️ Important Notes:**
- Requires SerpAPI API key set as `SERPAPI_KEY` environment variable
- Round trip flights (type=1) require `return_date` parameter
- Rate limits apply based on your SerpAPI plan
- See [SerpAPI Google Flights](https://serpapi.com/google-flights-api) for detailed documentation


---

## 👥 Social Features Endpoints

### Create Community
```http
POST /communities
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "name": "Rajasthan Travelers",
  "description": "A community for travelers exploring Rajasthan"
}
```

### Create Post
```http
POST /posts
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "content": "Just visited the amazing City Palace in Udaipur!",
  "community_id": "community_object_id_here"
}
```

### Like Post
```http
POST /posts/{post_id}/like
Authorization: Bearer {access_token}
```

### Comment on Post
```http
POST /posts/{post_id}/comment
Authorization: Bearer {access_token}
Content-Type: application/json

{
  "content": "Looks amazing! How was your experience?"
}
```

### Share Post
```http
POST /posts/{post_id}/share
Authorization: Bearer {access_token}
```

### Get Feed
```http
GET /feed?skip=0&limit=10
Authorization: Bearer {access_token}
```

**Query Parameters:**
- `skip` (int): Number of posts to skip (pagination)
- `limit` (int): Number of posts to return (max 10)

**Response:**
```json
{
  "feed": [
    {
      "_id": "post_id",
      "content": "Post content",
      "author": "user@example.com",
      "created_at": "2025-11-26T10:00:00",
      "likes": ["user1@email.com", "user2@email.com"],
      "comments": ["comment_id_1", "comment_id_2"],
      "shares": 5
    }
  ]
}
```

### Get My Posts
```http
GET /my-posts
Authorization: Bearer {access_token}
```

---

## 🔧 Utility Endpoints

### Contact Us
```http
POST /contactus
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "message": "I have a question about..."
}
```

### Report Bug
```http
GET /reportbug?message=Description of the bug
```

### Health Check
```http
GET /
```

**Response:**
```json
{
  "message": "Hello World"
}
```

---

## 🔑 Admin Endpoints

### Create User (Admin Only)
```http
POST /admin/create-user
Content-Type: application/json

{
  "email": "newuser@example.com",
  "password": "password123",
  "first_name": "Jane",
  "last_name": "Smith",
  "secret": "backpackkRogueported"
}
```

⚠️ Requires admin secret key.

---

## 📊 GraphQL Endpoint

### GraphQL Queries
```http
POST /graphql
Content-Type: application/json

{
  "query": "{ trips { id cityName noDays } }"
}
```

GraphiQL interface available at: `http://localhost:8000/graphql`

---

## 🔒 Authentication

Most endpoints require Bearer token authentication:

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

Get token from `/token` endpoint after login.

---

## 📝 Common HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing/invalid token |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource doesn't exist |
| 500 | Internal Server Error |

---

## 💡 Usage Tips

1. **Always verify email** after registration before generating trips
2. **Cache place data**: Use `/getplaces` once, then use cached data
3. **Optimize photo requests**: Set `include_photos: false` if not needed
4. **Choose appropriate travel schedule**:
   - Leisure for families/relaxed trips
   - Explorer for balanced itineraries
   - Adventurer for packed schedules
5. **Use autocomplete** for city names to ensure correct formatting

---

## 🚀 Quick Start Example

```bash
# 1. Register
curl -X POST http://localhost:8000/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"pass123","first_name":"John","last_name":"Doe"}'

# 2. Verify email (check inbox for link)

# 3. Login
curl -X POST http://localhost:8000/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=pass123"

# 4. Generate trip (use token from step 3)
curl -X POST http://localhost:8000/tripgenerator \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "city_name": "Udaipur, Rajasthan, India",
    "place_types": ["museum", "tourist_attraction"],
    "travel_schedule": "Explorer",
    "no_of_days": 2,
    "weekdays": ["Monday", "Tuesday"]
  }'
```

---

## 📚 Related Documentation

- [PROJECT_INDEX.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/PROJECT_INDEX.md) - Complete project documentation
- [CONTINUOUS_STRUCTURE_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/CONTINUOUS_STRUCTURE_README.md) - Trip structure details
- [TRIP_MODIFICATION_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/TRIP_MODIFICATION_README.md) - Trip modification guide

---

**API Version**: 1.0  
**Last Updated**: November 26, 2025
