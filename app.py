"""
ENHANCED UK Visa Sponsorship Assistant with Semantic Understanding
Intelligently finds tech companies even without "technology" in their name

New Features:
- Web enrichment to determine company industries
- LLM-powered filtering for better relevance
- Semantic understanding (e.g., JPMorgan = tech-heavy finance)
- Combined Excel data + web search results
"""
import os
import traceback
from pathlib import Path

from config import (
    EMBEDDINGS_CACHE_FILE,
    DATA_CACHE_FILE,
    clear_cache
)
from vector_db import EnhancedVectorDatabase
from agent import AgenticVisaAssistant


def main():
    """Main entry point for the Enhanced UK Visa Assistant"""
    print("\n" + "="*80)
    print("🤖 UK VISA SPONSORSHIP ASSISTANT")
    print("   with Universal Semantic Understanding")
    print("="*80)
    print("✨ Features:")
    print("   • Semantic search across ALL industries (tech, finance, healthcare, etc.)")
    print("   • Web enrichment for accurate industry detection")
    print("   • LLM-powered intelligent result filtering")
    print("   • Combined Excel + web search results")
    print("="*80 + "\n")

    # Get API key
    gemini_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not gemini_api_key:
        print("🔑 Enter Google Gemini API Key")
        print("Get free key: https://aistudio.google.com/app/apikey\n")
        gemini_api_key = input("API Key: ").strip()
        if not gemini_api_key:
            print("❌ Gemini API key required to continue")
            return

    # Get CSV file path
    print("\n📂 UK Sponsor Database Setup")
    print("-" * 80)
    print("Download the official register from:")
    print("https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers")
    print("-" * 80)

    csv_path = None

    # Check if cached data exists
    if EMBEDDINGS_CACHE_FILE.exists() and DATA_CACHE_FILE.exists():
        print("\n✅ Found cached database")
        use_cache = input("Use cached database? (Y/n): ").strip().lower()
        if use_cache != 'n':
            csv_path = None
        else:
            csv_path = input("\nEnter path to CSV file: ").strip()
            # Ask if user wants enrichment
            if csv_path:
                print("\n⚠️  Web enrichment will make initial indexing slower but improves results")
                enrich = input("Enable web enrichment for top companies? (Y/n): ").strip().lower()
                if enrich == 'n':
                    print("   Skipping enrichment - using name-based tech detection only")
    else:
        csv_path = input("\nEnter path to CSV file: ").strip()

    if csv_path and not Path(csv_path).exists():
        print(f"❌ File not found: {csv_path}")
        return

    # Initialize Gemini model first (needed for LLM filtering)
    import google.generativeai as genai
    genai.configure(api_key=gemini_api_key)
    from config import GEMINI_MODEL_NAME
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    # Initialize enhanced vector database
    print("\n🚀 Initializing Enhanced Vector Database...\n")
    try:
        vector_db = EnhancedVectorDatabase(csv_path, gemini_model)
    except Exception as e:
        print(f"❌ Vector database initialization failed: {e}")
        traceback.print_exc()
        return

    # Check if database loaded
    if vector_db.sponsors_df is None or len(vector_db.sponsors_df) == 0:
        print("\n❌ No sponsor data loaded. Please provide a valid CSV file.")
        return

    # Initialize agent
    print("\n🚀 Initializing agentic assistant with Gemini...\n")
    try:
        agent = AgenticVisaAssistant(gemini_api_key, vector_db)
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        traceback.print_exc()
        return

    # Show stats
    stats = agent.get_stats()
    print("\n📊 Database Statistics:")
    print(f"   • Total Sponsors: {stats.get('total_sponsors', 0):,}")
    print(f"   • A-Rated (Active): {stats.get('a_rated', 0):,}")
    print(f"   • Web-Enriched Companies: {stats.get('enriched_companies', 0):,}")
    print(f"   • Companies with Industry Data: {stats.get('companies_with_industries', 0):,}")
    print(f"   • Unique Cities: {stats.get('unique_cities', 0):,}")
    print(f"   • Model: {stats.get('model', 'Unknown')}")

    print("\n" + "="*80)
    print("💬 READY FOR INTELLIGENT INTERACTION")
    print("="*80)
    print("\n🤖 Universal Semantic Search:")
    print("   • Understands ALL industries (not just tech)")
    print("   • Matches companies by actual business, not just name")
    print("   • Combines Excel data + web verification")
    print("   • LLM filters results for best matches")
    print("\n📝 Try queries across ANY industry:")
    print("   • 'Find technology companies in London'")
    print("     → JPMorgan, Goldman Sachs, Microsoft, Accenture...")
    print("   • 'Healthcare and pharmaceutical companies'")
    print("     → NHS Trusts, pharma companies, biotech firms...")
    print("   • 'Finance and banking companies'")
    print("     → Banks, investment firms, insurance...")
    print("   • 'Consulting firms'")
    print("     → McKinsey, Deloitte, PwC, Accenture...")
    print("   • 'Retail and e-commerce companies'")
    print("     → Amazon, retailers, fashion brands...")
    print("\n⌨️  Commands:")
    print("   • 'stats' - View system statistics")
    print("   • 'clear' - Clear conversation history")
    print("   • 'clearcache' - Clear database cache and rebuild")
    print("   • 'quit' - Exit")
    print("="*80 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("👤 You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Good luck with your visa application and job search!\n")
                break

            if user_input.lower() == 'stats':
                stats = agent.get_stats()
                print("\n📊 System Statistics:")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                print()
                continue

            if user_input.lower() in ['clear', 'reset']:
                from models import AgentMemory
                agent.memory = AgentMemory()
                print("\n🔄 Conversation cleared!\n")
                continue

            if user_input.lower() == 'clearcache':
                print("\n🗑️  Clearing cache and rebuilding database...")
                clear_cache()
                csv_path = input("Enter path to CSV file: ").strip()
                if csv_path and Path(csv_path).exists():
                    print("⚠️  Rebuilding with web enrichment...")
                    vector_db.load_and_index_csv(csv_path, enrich_top_n=100)
                    print("✅ Database rebuilt successfully!\n")
                else:
                    print("❌ Invalid CSV path. Cache cleared but database not rebuilt.\n")
                continue

            # Process with agentic loop
            response = agent.chat(user_input)
            print(f"\n🤖 Assistant:\n{response}\n")
            print("="*80 + "\n")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            traceback.print_exc()


if __name__ == "__main__":
    main()
