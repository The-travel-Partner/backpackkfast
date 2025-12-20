"""
RAG-based Agent using CrewAI with JSON Database and SerpAPI Google Search

This module implements a Retrieval-Augmented Generation (RAG) agent that can:
1. Search a local JSON knowledge base for information
2. Perform Google searches using SerpAPI
3. Intelligently combine information from both sources
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Type

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from serpapi import GoogleSearch


class JSONSearchInput(BaseModel):
    """Input schema for JSON Search Tool"""
    query: str = Field(..., description="Search query to find relevant information in the JSON database")


class JSONSearchTool(BaseTool):
    """Custom tool to search through a JSON knowledge base"""
    
    name: str = "JSON Knowledge Base Search"
    description: str = (
        "Searches the local JSON knowledge base for travel-related information. "
        "Use this tool to find information about destinations, hotels, activities, "
        "travel tips, and cuisine that are stored in our database. "
        "This is useful for getting curated, verified information from our knowledge base."
    )
    args_schema: Type[BaseModel] = JSONSearchInput
    json_file_path: str = Field(default="")
    
    def __init__(self, json_file_path: str = None):
        """Initialize the JSON Search Tool"""
        if json_file_path is None:
            # Default to knowledge_base.json in the same directory
            current_dir = Path(__file__).parent
            json_file_path = str(current_dir / "knowledge_base.json")
        
        super().__init__(json_file_path=json_file_path)
    
    def _run(self, query: str) -> str:
        """Execute the search in the JSON database"""
        try:
            # Load JSON data
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            query_lower = query.lower()
            results = []
            
            # Search through all entries
            for item in data.get('travel_info', []):
                item_str = json.dumps(item).lower()
                
                # Check if query terms appear in the item
                if query_lower in item_str:
                    results.append(item)
                    continue
                
                # Also check individual words
                query_words = query_lower.split()
                matches = sum(1 for word in query_words if word in item_str)
                if matches >= len(query_words) * 0.5:  # At least 50% of words match
                    results.append(item)
            
            if not results:
                return f"No information found in the knowledge base for query: '{query}'"
            
            # Format results
            formatted_results = []
            for idx, item in enumerate(results[:5], 1):  # Limit to top 5 results
                formatted_results.append(f"\n--- Result {idx} ---")
                for key, value in item.items():
                    formatted_results.append(f"{key}: {value}")
            
            return "\n".join(formatted_results)
            
        except FileNotFoundError:
            return f"Error: Knowledge base file not found at {self.json_file_path}"
        except json.JSONDecodeError:
            return "Error: Invalid JSON format in knowledge base"
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"


class GoogleSearchInput(BaseModel):
    """Input schema for Google Search Tool"""
    query: str = Field(..., description="Search query to find information on Google")


class SerpAPISearchTool(BaseTool):
    """Custom tool to perform Google searches using SerpAPI"""
    
    name: str = "Google Search"
    description: str = (
        "Performs a Google search using SerpAPI to find current, real-time information "
        "from the internet. Use this tool when you need up-to-date information, "
        "recent news, reviews, or information not available in the knowledge base. "
        "Requires SERPAPI_API_KEY environment variable to be set."
    )
    args_schema: Type[BaseModel] = GoogleSearchInput
    
    def _run(self, query: str) -> str:
        """Execute Google search using SerpAPI"""
        try:
            api_key = os.getenv('SERPAPI_API_KEY')
            if not api_key:
                return (
                    "Error: SERPAPI_API_KEY not found in environment variables. "
                    "Please set it in your .env file. Get your free API key at https://serpapi.com/"
                )
            
            # Perform search
            search = GoogleSearch({
                "q": query,
                "api_key": api_key,
                "num": 10  # Get top 5 results
            })
            
            results = search.get_dict()
            
            # Extract organic results
            organic_results = results.get('organic_results', [])
            
            if not organic_results:
                return f"No search results found for query: '{query}'"
            
            # Format results
            formatted_results = [f"Google Search Results for: '{query}'\n"]
            for idx, result in enumerate(organic_results[:5], 1):
                title = result.get('title', 'N/A')
                snippet = result.get('snippet', 'N/A')
                link = result.get('link', 'N/A')
                
                formatted_results.append(f"\n--- Result {idx} ---")
                formatted_results.append(f"Title: {title}")
                formatted_results.append(f"Snippet: {snippet}")
                formatted_results.append(f"Link: {link}")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            return f"Error performing Google search: {str(e)}"


class RAGAgent:
    """RAG-based Agent that combines JSON database and Google search"""
    
    def __init__(self, json_file_path: str = None, google_api_key: str = None):
        """
        Initialize the RAG Agent
        
        Args:
            json_file_path: Path to JSON knowledge base file
            google_api_key: Google API key for Gemini (defaults to env variable)
        """
        # Initialize tools
        self.json_tool = JSONSearchTool(json_file_path=json_file_path)
        self.search_tool = SerpAPISearchTool()
        
        # Initialize LLM
        api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file or pass it to the constructor."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.7
        )
        
        # Create agent
        self.agent = Agent(
            role='Travel Information Specialist',
            goal='Provide accurate and comprehensive travel information by searching both local knowledge base and the internet',
            backstory=(
                "You are an expert travel information specialist with access to a curated "
                "knowledge base of travel information and the ability to search the internet "
                "for real-time updates. You intelligently decide which source to use based on "
                "the query - using the knowledge base for established information and Google "
                "search for current events, reviews, and latest updates."
            ),
            tools=[self.json_tool, self.search_tool],
            llm=self.llm,
            verbose=True,
            allow_delegation=False
        )
    
    def query(self, question: str) -> str:
        """
        Query the RAG agent with a question
        
        Args:
            question: The question to ask the agent
            
        Returns:
            The agent's response
        """
        # Create task
        task = Task(
            description=f"Answer the following question: {question}",
            agent=self.agent,
            expected_output="A comprehensive answer to the question using information from the knowledge base and/or Google search"
        )
        
        # Create crew and execute
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True
        )
        
        result = crew.kickoff()
        return str(result)


def test_serpapi():
    """Test function to verify SerpAPI is working"""
    tool = SerpAPISearchTool()
    result = tool._run("best time to visit India")
    print(result)
    return result


def main():
    """Example usage of the RAG Agent"""
    print("Initializing RAG Agent...")
    
    try:
        # Initialize agent
        agent = RAGAgent()
        
        # Example queries
        queries = [
            "What information do you have about Goa?",
            "What are the latest hotel reviews for Taj Hotels?",
            "Tell me about travel tips in your database",
        ]
        
        for query in queries:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print('='*80)
            
            response = agent.query(query)
            print(f"\nResponse:\n{response}\n")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("- GOOGLE_API_KEY (for Gemini)")
        print("- SERPAPI_API_KEY (for Google Search)")


if __name__ == "__main__":
    main()
