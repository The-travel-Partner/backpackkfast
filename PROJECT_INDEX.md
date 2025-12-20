# Backpackk Project Index

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Core Modules](#core-modules)
- [API Endpoints](#api-endpoints)
- [Database Collections](#database-collections)
- [Configuration](#configuration)
- [Data Models](#data-models)

---

## 🎯 Project Overview

**Backpackk** is a comprehensive AI-powered travel planning platform built with FastAPI. It provides intelligent trip generation, itinerary management, and social features for travelers.

### Key Features
- 🗺️ **AI Trip Generation**: Automated itinerary creation using Google Places API and Vertex AI
- 🔐 **Authentication**: Email/password and Google OAuth2 authentication
- 📍 **Place Discovery**: Smart place retrieval with Redis caching
- ⏰ **Time Optimization**: Intelligent schedule calculation based on travel preferences
- 🌐 **Social Features**: Community posts, comments, likes, and sharing
- 📊 **GraphQL Support**: Alternative query interface via Strawberry GraphQL
- 🖼️ **Photo Management**: Automated photo retrieval and storage for places

---

## 💻 Technology Stack

### Backend
- **Framework**: FastAPI (async Python web framework)
- **Language**: Python 3.x
- **API Standard**: REST + GraphQL

### Database
- **Primary DB**: MongoDB (via Motor AsyncIO client)
- **Caching**: Redis (with fallback to MongoDB)
- **Connection String**: MongoDB Atlas cloud cluster

### AI & External APIs
- **AI Platform**: Google Vertex AI (Gemini 2.5 Flash)
- **Maps & Places**: Google Maps API, Google Places API (New)
- **OAuth**: Google OAuth2
- **Email**: SMTP (Gmail)

### Deployment
- **Platform**: Google Cloud Run
- **Container**: Docker
- **Alternative**: Heroku (via Procfile)

---

## 📁 Project Structure

```
backpackkfast/
├── main.py                           # Main FastAPI application (1556 lines)
├── config.py                         # Configuration and shared constants
├── models.py                         # Pydantic models for all collections
├── graphql_schema.py                 # GraphQL schema definitions
├── Redis.py                          # Redis manager with fallback logic
├── trip_time_recalculator.py        # Trip timing optimization engine
│
├── authenticate/                     # Authentication module
│   ├── authentication.py             # JWT & user authentication logic
│   └── verifytempToken.py           # Temporary token verification
│
├── tripgen/                          # Trip generation core
│   ├── tripcreator.py               # Main trip creation orchestrator
│   ├── placesRetrievenew.py         # Place retrieval with AI scoring
│   ├── asyncclass.py                # Async utilities for place/photo fetching
│   ├── bestplacesModel.py           # Best places selection model
│   ├── getplacesModel.py            # Place retrieval request model
│   ├── tripgenModel.py              # Trip generation request models
│   └── placesDBClass.py             # Database interaction for places
│
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── app.yaml                          # Google App Engine config
├── Procfile                          # Heroku deployment config
│
├── CONTINUOUS_STRUCTURE_README.md    # Trip structure documentation
├── TRIP_MODIFICATION_README.md       # Trip modification API docs
└── README.md                         # Project overview

Data Files:
├── bigdata.csv                       # Place dataset
├── df_sorted.csv                     # Sorted places data
└── jaipurdata.json                   # Jaipur city places data
```

---

## 🔧 Core Modules

### 1. **main.py** (1556 lines)
The central FastAPI application containing all API endpoints.

**Key Sections:**
- Lines 1-96: Imports, initialization, middleware setup
- Lines 99-228: Authentication endpoints (login, register, verify)
- Lines 229-241: Trip generator endpoint
- Lines 245-487: OAuth2 Google authentication flow
- Lines 495-568: Place retrieval and caching
- Lines 570-656: Places data endpoints
- Lines 657-796: Photo retrieval endpoints
- Lines 802-936: AI place descriptions (Gemini)
- Lines 959-1127: Place details and nearby places
- Lines 1129-1318: Advanced place search
- Lines 1327-1368: Admin user creation
- Lines 1374-1471: Social features (communities, posts, comments)
- Lines 1474-1556: Trip modification with time recalculation

### 2. **config.py**
Centralized configuration to avoid circular imports.

```python
# Database
mongostr = "mongodb+srv://..."
client = AsyncIOMotorClient(mongostr)
db = client['backpackk']
usercollection = db['users']

# Authentication
SECRET_KEY = "..."
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 43800  # ~30 days

# API Keys
apikey = "AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk"
```

### 3. **models.py** (465 lines)
Pydantic models with BSON ObjectId handling for all collections.

**Model Categories:**
- Trip models: `TripData`, `TripInDB`, `DayItinerary`, `PlaceDetails`
- Community models: `CommunityBase`, `CommunityCreate`, `CommunityInDB`
- Post models: `PostBase`, `PostCreate`, `CommentCreate`
- Utility models: `PaginatedResponse`, `PyObjectId`

### 4. **tripgen/tripcreator.py**
Core trip generation orchestrator using continuous structure.

**Key Features:**
- Continuous place numbering (no separate meal sections)
- Travel schedule support (Leisure, Explorer, Adventurer)
- Optimized time slot calculation
- Redis caching with MongoDB fallback

### 5. **trip_time_recalculator.py** (40,039 bytes)
Advanced time optimization engine for trip modifications.

**Capabilities:**
- Recalculates timing for rearranged places
- Uses OlaMaps API with Haversine fallback
- Supports both old (wrapper) and new (continuous) structures
- Travel duration calculation based on coordinates

### 6. **Redis.py**
Redis manager with graceful error handling.

```python
class RedisManager:
    def get(key) -> Optional[str]
    def setex(key, time, value) -> bool
    # Returns None/False on failure, continues without crash
```

### 7. **graphql_schema.py**
Strawberry GraphQL schema for alternative querying.

**Available Queries:**
- User queries
- Trip queries
- Community queries
- Post queries

---

## 🌐 API Endpoints

### Authentication & User Management
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/register` | User registration with email verification | ❌ |
| POST | `/token` | Login and get access token | ❌ |
| GET | `/verify/{user_id}/{token}` | Email verification | ❌ |
| GET | `/users/me` | Get current user profile | ✅ |
| GET | `/google` | Google OAuth2 authentication | ❌ |
| GET | `/callback` | Google OAuth2 callback | ❌ |
| POST | `/verifytemp` | Verify temporary token | ❌ |
| POST | `/forgot-password` | Request password reset | ❌ |
| POST | `/reset-password` | Reset password | ❌ |
| POST | `/admin/create-user` | Admin user creation | 🔑 Secret |

### Trip Generation
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/tripgenerator` | Generate new trip itinerary | ✅ |
| POST | `/number/of/available/days` | Calculate available trip days | ✅ |
| POST | `/trips/modify` | Modify trip with time recalculation | ✅ |
| GET | `/gettrips` | Get user's trips (latest 3 or all) | ✅ |

### Places & Photos
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/getplaces` | Retrieve places for a city | ✅ |
| GET | `/getplaces/all` | Get all places for a city | ✅ |
| GET | `/autocomplete` | City name autocomplete | ❌ |
| POST | `/getphotos` | Get photos by photo reference | ❌ |
| POST | `/getphotos/byplaceid` | Get photos by place ID | ❌ |
| GET | `/getphoto` | Get single photo | ✅ |
| POST | `/place` | Get detailed place information | ✅ |
| POST | `/nearby-places` | Find nearby attractions/restaurants | ✅ |
| GET | `/placedescription` | AI-generated place description | ✅ |

### Maps & Visualization
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/generate-map-blob` | Generate static map image URL | ✅ |
| GET | `/generate-map-blob/streetview` | Street view embed URL | ✅ |
| GET | `/generate-map-blob/view` | Place view embed URL | ✅ |

### Social Features
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| POST | `/communities` | Create community | ✅ |
| POST | `/posts` | Create post | ✅ |
| POST | `/posts/{post_id}/like` | Like a post | ✅ |
| POST | `/posts/{post_id}/comment` | Comment on post | ✅ |
| POST | `/posts/{post_id}/share` | Share post | ✅ |
| GET | `/feed` | Get social feed (paginated) | ✅ |
| GET | `/my-posts` | Get user's posts | ✅ |

### Utility
| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/` | Health check | ❌ |
| POST | `/contactus` | Contact form submission | ❌ |
| GET | `/reportbug` | Report bug | ❌ |
| POST | `/graphql` | GraphQL endpoint | Varies |

---

## 🗄️ Database Collections

### MongoDB Collections (`backpackk` database)

#### 1. **users**
User accounts and authentication data.
```python
{
    "_id": ObjectId,
    "email": str,
    "first_name": str,
    "last_name": str,
    "hashed_password": str,
    "disabled": bool,
    "verified": bool,
    "verification_token": str (optional)
}
```

#### 2. **user_trips**
User trip itineraries.
```python
{
    "_id": ObjectId,
    "email": str,
    "trips": [
        {
            "city_name": str,
            "place_types": [str],
            "no_of_days": int,
            "trip": {
                "Day 1": {
                    "Name": {"0": str, "1": str, ...},
                    "lat": {"0": float, ...},
                    "lng": {"0": float, ...},
                    "rating": {"0": float, ...},
                    "place_id": {"0": str, ...},
                    "type": {"0": str, ...},
                    "Weekday": str,
                    "Slots": {...}
                }
            }
        }
    ]
}
```

#### 3. **placesdata**
Cached place data for cities.
```python
{
    "_id": ObjectId,
    "city_name": str,
    "timestamp": str,
    "places": [
        {
            "name": str,
            "lat": float,
            "lng": float,
            "rating": float,
            "place_id": str,
            "type": str,
            ...
        }
    ]
}
```

#### 4. **placesData** (with capital D)
Individual place details with photos.
```python
{
    "_id": ObjectId,
    "place_id": str,
    "name": str,
    "place_type": str,
    "location": {"latitude": float, "longitude": float},
    "rating": float,
    "vicinity": str,
    "timestamp": datetime
}
```

#### 5. **images**
Place photos storage.
```python
{
    "_id": ObjectId,
    "place_id": str,
    "image": str (base64),
    "content_type": str,
    "timestamp": str
}
```

#### 6. **descriptions**
AI-generated place descriptions.
```python
{
    "_id": ObjectId,
    "place_name": str,
    "city_name": str,
    "description": str,
    "timestamp": datetime
}
```

#### 7. **communities**
Social communities.
```python
{
    "_id": ObjectId,
    "name": str,
    "description": str,
    "creator": str (email),
    "members": [str],
    "created_at": datetime
}
```

#### 8. **posts**
Social posts.
```python
{
    "_id": ObjectId,
    "content": str,
    "community_id": str (optional),
    "author": str (email),
    "created_at": datetime,
    "likes": [str] (emails),
    "comments": [str] (comment IDs),
    "shares": int
}
```

#### 9. **comments**
Post comments.
```python
{
    "_id": ObjectId,
    "post_id": str,
    "author": str (email),
    "content": str,
    "created_at": datetime
}
```

#### 10. **message**
Contact form submissions.
```python
{
    "_id": ObjectId,
    "name": str,
    "email": str,
    "message": str
}
```

#### 11. **bugreport**
Bug reports.
```python
{
    "_id": ObjectId,
    "message": str
}
```

### Redis Cache Keys

- **User places cache**: `{user_email}` → places data (1 hour TTL)
- **Place descriptions**: `description:{city_name}:{place_name}` → description (7 days TTL)

---

## ⚙️ Configuration

### Environment Variables
The project uses hardcoded values (should be migrated to environment variables):

```bash
# Database
MONGODB_URI=mongodb+srv://backpackkuser:7ZSTOXH06gjhUvl8@cluster0.vehwdql.mongodb.net/

# Authentication
SECRET_KEY=83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=43800

# Google APIs
GOOGLE_API_KEY=AIzaSyAt9_35pEEtevoHJCTeJwynPqjx-9-MVjk
GOOGLE_CLIENT_ID=794713488480-8iqh9m6p3a93clvqrfrdjakt8q22movg.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=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

# Email (Gmail SMTP)
SENDER_EMAIL=nimishspslosal@gmail.com
SENDER_PASSWORD=axmdvfpmiewtzmsd

# Deployment
ORIGIN_URL=https://backpackk-cloud.el.r.appspot.com/
```

### CORS Configuration
```python
origins = [
    'https://backpackk.com',
    'http://localhost:4173',
    'http://localhost:5173',
    '*'  # Allows all origins
]
```

---

## 📊 Data Models

### Travel Schedules
| Schedule | Slots/Day | Duration | Pace | Best For |
|----------|-----------|----------|------|----------|
| **Leisure** | 3 | 45+ min | Relaxed | Families, couples |
| **Explorer** | 4 | 30+ min | Moderate | Most travelers |
| **Adventurer** | 5 | 25+ min | Fast | Active explorers |

### Place Types
**Attractions:**
- `tourist_attraction`
- `museum`
- `hindu_temple`
- `zoo`

**Dining:**
- `restaurant`
- `vegetarian_restaurant`

**Nightlife:**
- `bar`
- `night_club`

### Continuous Structure Format
Places are numbered sequentially with integrated meals:
```json
{
  "Day 1": {
    "Name": {
      "0": "Attraction 1",
      "1": "Attraction 2",
      "2": "Lunch Restaurant",
      "3": "Attraction 3",
      "4": "Dinner Restaurant",
      "5": "Bar/Club"
    },
    "type": {
      "0": "tourist_attraction",
      "1": "museum",
      "2": "vegetarian_restaurant",
      "3": "tourist_attraction",
      "4": "restaurant",
      "5": "bar"
    },
    "Weekday": "Tuesday"
  }
}
```

---

## 📚 Additional Documentation

For detailed information on specific topics:

- **[CONTINUOUS_STRUCTURE_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/CONTINUOUS_STRUCTURE_README.md)**: Continuous trip structure implementation details
- **[TRIP_MODIFICATION_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/TRIP_MODIFICATION_README.md)**: Trip modification API usage guide
- **[README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/README.md)**: Basic project overview

---

## 🚀 Quick Start

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment
```bash
# Build image
docker build -t backpackk-api .

# Run container
docker run -p 8000:8000 backpackk-api
```

### Google Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy backpackkfast --source .
```

---

## 🔑 Key Dependencies

```txt
fastapi              # Web framework
motor                # Async MongoDB driver
redis                # Redis client (optional)
pydantic             # Data validation
passlib[bcrypt]      # Password hashing
python-jose[cryptography]  # JWT tokens
google-genai         # Vertex AI SDK
googlemaps           # Google Maps API
strawberry-graphql[fastapi]  # GraphQL
aiohttp              # Async HTTP client
pandas               # Data manipulation
```

---

## 📝 Notes

### Security Concerns
⚠️ **WARNING**: The following should be addressed:
- API keys and secrets are hardcoded (should use environment variables)
- MongoDB credentials exposed in code
- Email credentials hardcoded
- No rate limiting implemented
- Admin secret key is weak

### Performance Optimizations
- Redis caching reduces MongoDB queries
- Async/await for concurrent operations
- Connection pooling via Motor
- Photo retrieval uses semaphores to limit concurrency

### Future Improvements
- Migrate secrets to environment variables
- Add rate limiting
- Implement proper logging
- Add unit tests
- Create API documentation (OpenAPI/Swagger)
- Add monitoring and observability
- Implement webhook support for real-time updates

---

**Last Updated**: November 26, 2025  
**Version**: 1.0  
**Maintainer**: Rogue
