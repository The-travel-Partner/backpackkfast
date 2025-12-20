# RAG-based Agent with CrewAI - Quick Start Guide

## Overview

A Retrieval-Augmented Generation (RAG) agent built with CrewAI that combines:
- **Local JSON Knowledge Base** - Curated travel information stored locally
- **Google Search via SerpAPI** - Real-time web search capabilities

The agent intelligently decides which tool to use based on your query.

## Setup

### 1. Install Dependencies

```powershell
pip install crewai crewai-tools langchain langchain-google-genai python-dotenv
```

Or install from requirements.txt:
```powershell
pip install -r requirements.txt
```

### 2. Configure API Keys

Add these to your `.env` file:

```env
# Google Gemini API Key (Required)
GOOGLE_API_KEY=your_google_api_key_here

# SerpAPI Key (Required for Google Search)
SERPAPI_API_KEY=your_serpapi_key_here
```

**Get API Keys:**
- Google Gemini: https://makersuite.google.com/app/apikey
- SerpAPI (Free tier available): https://serpapi.com/

### 3. Files Created

- `ai/agent.py` - Main RAG agent implementation
- `ai/knowledge_base.json` - Sample JSON database with travel info
- `ai/rag_agent_example.py` - Example usage scripts
- `ai/.env.template` - Environment variable template

## Usage

### Quick Test

```python
from ai.agent import RAGAgent

# Initialize agent
agent = RAGAgent()

# Ask a question
response = agent.query("What destinations are in your database?")
print(response)
```

### Run Examples

```powershell
cd c:\Users\Rogue\PycharmProjects\backpackkfast
python ai\rag_agent_example.py
```

The example script provides:
1. **JSON Knowledge Base queries** - Search local database
2. **Google Search queries** - Real-time web search
3. **Hybrid queries** - Combine both sources
4. **Test individual tools** - Test each tool separately
5. **Interactive mode** - Ask custom questions

### Using Individual Tools

#### JSON Search Tool

```python
from ai.agent import JSONSearchTool

tool = JSONSearchTool()
result = tool._run("hotels")
print(result)
```

#### SerpAPI Google Search Tool

```python
from ai.agent import SerpAPISearchTool

tool = SerpAPISearchTool()
result = tool._run("best travel destinations India")
print(result)
```

## How It Works

### RAG Architecture

```
User Query
    ↓
CrewAI Agent (Gemini 2.0)
    ↓
Decision: Which tool(s) to use?
    ↓
┌─────────────────┬──────────────────┐
│  JSON Search    │  Google Search   │
│  (Local KB)     │  (SerpAPI)       │
└─────────────────┴──────────────────┘
    ↓
Agent combines & synthesizes results
    ↓
Final Response
```

### When Each Tool Is Used

**JSON Knowledge Base** is used for:
- Information about destinations, hotels, activities in the database
- Travel tips and curated content
- Structured data queries

**Google Search** is used for:
- Real-time information (weather, news, restrictions)
- Recent reviews and ratings
- Current events and updates

## Customization

### Add Your Own Data

Edit `ai/knowledge_base.json` to add your travel data:

```json
{
  "travel_info": [
    {
      "id": 13,
      "category": "destination",
      "name": "Your Destination",
      "description": "...",
      "best_time_to_visit": "..."
    }
  ]
}
```

### Change the LLM

In `ai/agent.py`, modify the LLM initialization:

```python
self.llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Change model here
    google_api_key=api_key,
    temperature=0.7  # Adjust creativity (0.0-1.0)
)
```

### Customize Agent Behavior

Modify the agent's role, goal, and backstory in `RAGAgent.__init__()`:

```python
self.agent = Agent(
    role='Your Custom Role',
    goal='Your custom goal',
    backstory='Your custom backstory',
    tools=[self.json_tool, self.search_tool],
    llm=self.llm,
    verbose=True
)
```

## Example Queries

**JSON Database Queries:**
- "What destinations are in your database?"
- "Tell me about hotels you have information for"
- "What travel tips do you have?"

**Google Search Queries:**
- "Latest COVID travel restrictions for India"
- "Recent reviews of Taj Hotels"
- "Best time to visit Goa in 2025"

**Hybrid Queries:**
- "Tell me about Goa from your database and find recent tourist reviews"
- "Compare your hotel information with current online ratings"

## Troubleshooting

### Missing API Keys

Error: `GOOGLE_API_KEY not found`
- Add your Gemini API key to `.env` file

Error: `SERPAPI_API_KEY not found`
- Add your SerpAPI key to `.env` file

### Import Errors

Error: `ModuleNotFoundError: No module named 'crewai'`
- Run: `pip install crewai crewai-tools`

### JSON File Not Found

Error: `Knowledge base file not found`
- Ensure `ai/knowledge_base.json` exists
- Or specify custom path: `RAGAgent(json_file_path="path/to/your.json")`

## Next Steps

1. ✅ Add your API keys to `.env`
2. ✅ Customize `knowledge_base.json` with your data
3. ✅ Run `python ai\rag_agent_example.py` to see it in action
4. ✅ Integrate into your main application

## Integration with Your App

To use in your existing FastAPI app (`main.py`):

```python
from ai.agent import RAGAgent

# Initialize once (maybe in startup event)
rag_agent = RAGAgent()

@app.post("/api/rag-query")
async def query_rag_agent(query: str):
    response = rag_agent.query(query)
    return {"response": response}
```

Enjoy your RAG-based travel information agent! 🚀
