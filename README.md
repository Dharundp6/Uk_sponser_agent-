# UK Visa Sponsorship Assistant

AI-powered assistant for finding UK Skilled Worker visa sponsors with **universal semantic understanding** across ALL industries.

## Key Features

- 🧠 **Universal Semantic Search** - Understands companies across ALL industries (tech, finance, healthcare, retail, etc.)
- 🌐 **Web Enrichment** - Automatically detects company industries via Google
- 🤖 **LLM Filtering** - Gemini-powered intelligent result ranking
- 📊 **FAISS Vector DB** - Lightning-fast semantic similarity search
- 💬 **Agentic Chat** - Autonomous tool selection and multi-turn conversations
- 📄 **Resume Parsing** - Extract skills from PDF/DOCX and match jobs

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get API Key
Get a free Google Gemini API key: https://aistudio.google.com/app/apikey

### 3. Download Sponsor Data
Download the official UK sponsor register:
https://www.gov.uk/government/publications/register-of-licensed-sponsors-workers

### 4. Run the App
```bash
python app.py
```

Enter your API key when prompted, then provide the path to the CSV file.

## Project Structure

```
uk-rag/
├── app.py                  # Main application (CLI interface)
├── config.py               # Configuration and constants
├── models.py               # Data models (UserProfile, AgentMemory)
├── embeddings.py           # Embedding model operations
├── vector_db_enhanced.py   # Enhanced FAISS database with semantic search
├── tool_definitions.py     # Gemini function calling schemas
├── tools.py                # Tool implementations
├── agent.py                # Agentic orchestrator
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## How It Works

### Problem Solved
**Before**: Simple keyword matching only found companies with specific words in their name.

**After**: Universal semantic understanding finds companies across ALL industries based on their actual business.

### 3-Layer Intelligent Matching

#### 1. Web Enrichment
```
During indexing, for each company:
  → Google: "Company Name UK industry sector business"
  → Extract: Industry keywords from web results
  → Detect: All applicable industries (tech, finance, healthcare, etc.)
```

#### 2. Semantic Embeddings
```
Embed companies with industry context:
  Before: "JPMorgan Chase in London"
  After:  "JPMorgan Chase in London investment banking fintech digital technology"
  → Better semantic matching for ANY industry query
```

#### 3. LLM Filtering
```
Gemini intelligently ranks results based on query:
  Query: "technology companies" → Prioritize tech, fintech, software
  Query: "healthcare companies" → Prioritize medical, pharma, biotech
  Query: "finance companies" → Prioritize banking, investment, insurance
```

## Example Usage

### Find Technology Companies
```
You: Find technology companies in London

Assistant finds:
✓ JPMorgan Chase (fintech operations)
✓ Goldman Sachs (trading technology)
✓ Accenture (tech consulting)
✓ Deloitte (digital transformation)
✓ Microsoft UK
✓ IBM UK
```

### Find Healthcare Companies
```
You: Find healthcare and pharmaceutical companies

Assistant finds:
✓ NHS Trusts
✓ GlaxoSmithKline (pharma)
✓ AstraZeneca (biotech)
✓ Private hospitals
✓ Medical device companies
```

### Find Finance Companies
```
You: Find finance and banking companies in London

Assistant finds:
✓ Barclays
✓ HSBC
✓ JPMorgan
✓ Goldman Sachs
✓ Insurance companies
```

### Search for Jobs
```
You: Show me graduate jobs at Accenture

Assistant searches web and provides:
- Job listings with links
- Application deadlines
- Required skills
```

### Check Visa Eligibility
```
You: Am I eligible for Skilled Worker visa with £35,000 salary?

Assistant analyzes:
- Salary thresholds
- UK degree benefits
- Points requirements
- Application guidance
```

### Upload Resume
```
You: I want to upload my resume and find matching jobs

Assistant parses resume and:
- Extracts your skills
- Finds matching companies
- Ranks jobs by relevance

## Available Commands

- **stats** - View database statistics
- **clear** - Clear conversation history
- **clearcache** - Rebuild database with fresh data
- **quit** - Exit application

## Configuration

### Adjust Web Enrichment
Edit `vector_db_enhanced.py` line ~200:
```python
enrich_top_n=100  # Number of companies to enrich (default: 100)
```

### Customize Industry Keywords
Edit `vector_db.py` line ~49:
```python
self.industry_keywords = {
    'technology', 'finance', 'healthcare',
    'your_industry_keyword'  # Add custom industry terms
}
```

### Toggle LLM Filtering
If LLM filtering is slow or failing, edit `config.py`:
```python
USE_LLM_FILTERING = False  # Disable LLM (faster, still accurate via semantic search)
```

## Performance

### Indexing (First Time)
- Without enrichment: ~2 minutes for 15,000 companies
- With enrichment: ~10-15 minutes (enriches top 100)
- Subsequent runs: Instant (uses cache)

### Search Speed
- Semantic search: <100ms
- LLM filtering: 1-2 seconds
- Total query time: ~2 seconds

## Improvement Over Basic Search

**Query**: "Find technology companies in London"

| Metric | Basic Search | Enhanced Search | Improvement |
|--------|-------------|-----------------|-------------|
| Results | 5 companies | 13 companies | +160% |
| Major employers found | 0 | 6 (JPM, GS, etc.) | Significant |
| False positives | High | Low | Better precision |

## Technologies Used

- **Google Gemini** - LLM for agentic chat and filtering
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings (all-MiniLM-L6-v2)
- **BeautifulSoup** - Web scraping
- **PyPDF2 & python-docx** - Resume parsing
- **Pandas** - Data processing

## Troubleshooting

### LLM Filtering Errors?
**Symptom**: "LLM filtering failed" or "finish_reason is 2"

**Solution**: Disable LLM filtering in `config.py`:
```python
USE_LLM_FILTERING = False
```
Note: Semantic search alone is still very accurate! LLM just provides a slight boost.

### Web Enrichment Slow?
**Solution**: Reduce scope in `vector_db.py` line ~169:
```python
enrich_top_n=50  # Instead of 100
```

### Missing Companies in Results?
**Solution**: Add industry keywords in `vector_db.py` line ~49:
```python
self.industry_keywords = {
    # Add your industry-specific terms
    'your_industry', 'your_sector', ...
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or PR.

