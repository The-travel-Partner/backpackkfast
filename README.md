# Backpackk - AI-Powered Travel Planning Platform

FastAPI implementation of the Backpackk backend - an intelligent travel planning system that generates optimized itineraries using AI and Google Maps data.

## 🚀 Features

- **AI Trip Generation**: Automated itinerary creation with optimal timing and routing
- **Smart Place Discovery**: Intelligent place selection using Google Places API
- **Time Optimization**: Dynamic schedule calculation based on travel preferences
- **Social Platform**: Communities, posts, comments, and trip sharing
- **Multi-Auth**: Email/password and Google OAuth2 authentication
- **Photo Management**: Automated photo retrieval and caching
- **GraphQL Support**: Alternative query interface alongside REST API

## 📚 Documentation

- **[PROJECT_INDEX.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/PROJECT_INDEX.md)** - Complete project documentation including:
  - Project architecture and structure
  - All modules and components
  - Database schema (11 collections)
  - Configuration details
  - Data models and formats

- **[API_REFERENCE.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/API_REFERENCE.md)** - Quick API reference with:
  - 40+ endpoint examples with request/response
  - Authentication flows
  - Usage tips and best practices
  - Quick start guide

- **[CONTINUOUS_STRUCTURE_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/CONTINUOUS_STRUCTURE_README.md)** - Trip structure implementation details

- **[TRIP_MODIFICATION_README.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/TRIP_MODIFICATION_README.md)** - Trip modification API guide

## 🏗️ Architecture

```
FastAPI Application
├── Authentication (JWT + OAuth2)
├── Trip Generation Engine
│   ├── Place Discovery
│   ├── Time Optimization
│   └── Route Planning
├── Social Platform
│   ├── Communities
│   ├── Posts & Comments
│   └── Likes & Shares
└── Data Layer
    ├── MongoDB (Primary)
    └── Redis (Cache)
```

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Database**: MongoDB Atlas
- **Cache**: Redis
- **AI**: Google Vertex AI (Gemini)
- **APIs**: Google Maps, Google Places
- **Deployment**: Google Cloud Run / Heroku

## 📦 Installation

```bash
# Clone repository
git clone <repository-url>
cd backpackkfast

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## 🚢 Deployment

### Docker
```bash
docker build -t backpackk-api .
docker run -p 8000:8000 backpackk-api
```

### Google Cloud Run
```bash
gcloud run deploy backpackkfast --source .
```

## 📊 Key Statistics

- **Lines of Code**: 1,556 (main.py) + 465 (models.py) + tripgen modules
- **API Endpoints**: 40+ REST endpoints + GraphQL
- **Database Collections**: 11 MongoDB collections
- **Travel Schedules**: 3 types (Leisure, Explorer, Adventurer)
- **Supported Place Types**: 8+ categories (attractions, dining, nightlife)

## 🔑 Environment Variables

Create a `.env` file with:
```env
MONGODB_URI=your_mongodb_connection_string
SECRET_KEY=your_secret_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CLIENT_ID=your_oauth_client_id
GOOGLE_CLIENT_SECRET=your_oauth_client_secret
```

## 📖 Quick Start

```python
# 1. Register user
POST /register
{
  "email": "user@example.com",
  "password": "password",
  "first_name": "John",
  "last_name": "Doe"
}

# 2. Login
POST /token
username=user@example.com&password=password

# 3. Generate trip
POST /tripgenerator
Authorization: Bearer {token}
{
  "city_name": "Udaipur, Rajasthan, India",
  "place_types": ["museum", "tourist_attraction"],
  "travel_schedule": "Explorer",
  "no_of_days": 2
}
```

See [API_REFERENCE.md](file:///c:/Users/Rogue/PycharmProjects/backpackkfast/API_REFERENCE.md) for complete endpoint documentation.

## 🗂️ Project Structure

```
backpackkfast/
├── main.py                    # Main FastAPI application
├── config.py                  # Configuration
├── models.py                  # Pydantic models
├── Redis.py                   # Redis manager
├── trip_time_recalculator.py # Time optimization
├── authenticate/              # Authentication module
├── tripgen/                   # Trip generation core
└── [docs]                     # Documentation files
```

## 🤝 Contributing

Contributions welcome! Please check the documentation for architecture details before submitting PRs.

## 📝 License

[Add your license here]

## 👨‍💻 Maintainer

Rogue - [Contact Info]

---

**Last Updated**: November 26, 2025  
**Version**: 1.0
