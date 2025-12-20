"""
Example usage of the RAG-based Agent with CrewAI

This script demonstrates how to use the RAG agent with different types of queries:
1. Queries that use only the JSON knowledge base
2. Queries that use only Google search
3. Queries that benefit from both sources
"""

import os
from dotenv import load_dotenv
from ai.agent import RAGAgent, JSONSearchTool, SerpAPISearchTool

# Load environment variables
load_dotenv()


def example_json_queries():
    """Examples of queries that primarily use the JSON knowledge base"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Using JSON Knowledge Base")
    print("="*80)
    
    agent = RAGAgent()
    
    queries = [
        "What destinations are in your database for India?",
        "Tell me about hotels in your knowledge base",
        "What travel tips do you have about currency and payments?",
        "What activities can I do for adventure travel?",
    ]
    
    for query in queries:
        print(f"\n📝 Query: {query}")
        print("-" * 80)
        response = agent.query(query)
        print(f"Response: {response}\n")


def example_google_queries():
    """Examples of queries that primarily use Google search"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Using Google Search (SerpAPI)")
    print("="*80)
    
    agent = RAGAgent()
    
    queries = [
        "What are the current COVID-19 travel restrictions for India?",
        "Latest news about tourism in Goa",
        "Best rated restaurants in Jaipur 2025",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 80)
        response = agent.query(query)
        print(f"Response: {response}\n")


def example_hybrid_queries():
    """Examples of queries that benefit from both sources"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Hybrid Queries (JSON + Google Search)")
    print("="*80)
    
    agent = RAGAgent()
    
    queries = [
        "Tell me about Goa from your database and find recent tourist reviews online",
        "What hotels do you have information about, and what are their latest reviews?",
        "Compare the travel tips in your database with current travel advisories online",
    ]
    
    for query in queries:
        print(f"\n🔄 Query: {query}")
        print("-" * 80)
        response = agent.query(query)
        print(f"Response: {response}\n")


def test_individual_tools():
    """Test individual tools separately"""
    print("\n" + "="*80)
    print("TESTING INDIVIDUAL TOOLS")
    print("="*80)
    
    # Test JSON Search Tool
    print("\n📚 Testing JSON Search Tool...")
    json_tool = JSONSearchTool()
    result = json_tool._run("hotel")
    print(f"Result:\n{result}\n")
    
    # Test SerpAPI Tool
    print("\n🌐 Testing SerpAPI Google Search Tool...")
    search_tool = SerpAPISearchTool()
    result = search_tool._run("best travel destinations in India")
    print(f"Result:\n{result}\n")


def interactive_mode():
    """Interactive mode to ask custom questions"""
    print("\n" + "="*80)
    print("INTERACTIVE MODE")
    print("="*80)
    print("Ask questions to the RAG agent. Type 'exit' to quit.\n")
    
    agent = RAGAgent()
    
    while True:
        query = input("\n💬 Your question: ").strip()
        
        if query.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
        
        if not query:
            continue
        
        print("-" * 80)
        response = agent.query(query)
        print(f"\n🤖 Agent Response:\n{response}\n")


def main():
    """Main function to run examples"""
    print("🚀 RAG-based Agent Examples with CrewAI")
    print("=" * 80)
    
    # Check for required environment variables
    if not os.getenv('GOOGLE_API_KEY'):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables")
        print("Please add it to your .env file")
        return
    
    if not os.getenv('SERPAPI_API_KEY'):
        print("⚠️  Warning: SERPAPI_API_KEY not found in environment variables")
        print("Google search functionality will not work")
        print("Get your free API key at https://serpapi.com/\n")
    
    # Menu
    print("\nChoose an example to run:")
    print("1. JSON Knowledge Base queries")
    print("2. Google Search queries")
    print("3. Hybrid queries (both sources)")
    print("4. Test individual tools")
    print("5. Interactive mode")
    print("6. Run all examples")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == '1':
        example_json_queries()
    elif choice == '2':
        example_google_queries()
    elif choice == '3':
        example_hybrid_queries()
    elif choice == '4':
        test_individual_tools()
    elif choice == '5':
        interactive_mode()
    elif choice == '6':
        test_individual_tools()
        example_json_queries()
        example_google_queries()
        example_hybrid_queries()
    else:
        print("Invalid choice. Please run again and select 1-6.")


if __name__ == "__main__":
    main()
