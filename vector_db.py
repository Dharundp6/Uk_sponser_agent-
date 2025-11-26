"""
Enhanced FAISS vector database with semantic understanding
Includes web enrichment for better company-industry matching
"""
import faiss
import pandas as pd
import pickle
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from io import StringIO
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
import google.generativeai as genai

from embeddings import EmbeddingModel
from config import (
    FAISS_INDEX_FILE,
    EMBEDDINGS_CACHE_FILE,
    DATA_CACHE_FILE,
    METADATA_CACHE_FILE,
    EMBEDDING_DIMENSION
)


class EnhancedVectorDatabase:
    """
    Enhanced FAISS-based vector database with semantic understanding
    Combines CSV data + web enrichment for better industry matching
    """

    def __init__(self, csv_path: Optional[str] = None, gemini_model=None):
        """
        Initialize the enhanced vector database

        Args:
            csv_path: Optional path to CSV file containing sponsor data
            gemini_model: Gemini model instance for intelligent filtering
        """
        self.embedding_model = EmbeddingModel()
        self.faiss_index = None
        self.id_to_metadata = {}
        self.sponsors_df = None
        self.dimension = EMBEDDING_DIMENSION
        self.gemini_model = gemini_model

        # Industry keywords for comprehensive detection
        # Organized by sector for better semantic understanding
        self.industry_keywords = {
            # Technology
            'technology', 'tech', 'software', 'it services', 'information technology',
            'fintech', 'financial technology', 'digital', 'cloud', 'ai', 'data',
            'cybersecurity', 'blockchain', 'saas', 'platform', 'engineering',

            # Finance & Banking
            'finance', 'banking', 'investment', 'financial services', 'insurance',
            'asset management', 'wealth management', 'trading', 'capital markets',

            # Healthcare
            'healthcare', 'medical', 'pharmaceutical', 'biotech', 'hospital',
            'clinical', 'health services', 'life sciences', 'diagnostics',

            # Consulting
            'consulting', 'advisory', 'professional services', 'management consulting',
            'strategy', 'business consulting',

            # Retail & E-commerce
            'retail', 'e-commerce', 'consumer goods', 'fashion', 'luxury',

            # Manufacturing & Engineering
            'manufacturing', 'automotive', 'aerospace', 'construction', 'industrial',

            # Energy & Utilities
            'energy', 'oil', 'gas', 'renewable', 'utilities', 'power',

            # Media & Entertainment
            'media', 'entertainment', 'broadcasting', 'publishing', 'advertising',

            # Education
            'education', 'university', 'learning', 'training', 'academic',

            # Legal
            'legal', 'law', 'law firm', 'solicitors', 'barristers'
        }

        self._initialize_database(csv_path)

    def _initialize_database(self, csv_path: Optional[str] = None):
        """Initialize FAISS vector database from cache or CSV"""
        print("🔄 Initializing Enhanced FAISS database...")

        # Try to load from cache first
        if self._load_from_cache():
            return

        # If CSV path provided, load and index
        if csv_path and Path(csv_path).exists():
            print(f"📂 Loading CSV from: {csv_path}")
            self.load_and_index_csv(csv_path)
        else:
            print("⚠️  No cached data or CSV file provided.")
            print("    Please provide CSV path during initialization.")

    def _load_from_cache(self) -> bool:
        """Load database from cache files"""
        if FAISS_INDEX_FILE.exists() and METADATA_CACHE_FILE.exists() and DATA_CACHE_FILE.exists():
            print("📂 Loading from cache...")
            try:
                self.faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))

                with open(METADATA_CACHE_FILE, 'rb') as f:
                    self.id_to_metadata = pickle.load(f)

                with open(DATA_CACHE_FILE, 'rb') as f:
                    content = pickle.load(f)
                self.sponsors_df = pd.read_csv(StringIO(content.decode('utf-8')))

                print(f"✅ Loaded {self.faiss_index.ntotal} sponsors from cache")
                return True
            except Exception as e:
                print(f"⚠️  Cache load failed: {e}")
                return False
        return False

    def _enrich_company_data(self, company_name: str) -> Dict[str, Any]:
        """
        Enrich company data with web search to determine industry/sector

        Args:
            company_name: Name of the company

        Returns:
            Dictionary with industry information and detected sectors
        """
        try:
            query = f"{company_name} UK industry sector business"
            search_url = f"https://www.google.com/search?q={quote_plus(query)}&num=3"

            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(search_url, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract snippets
            snippets = []
            for result in soup.find_all('div', class_=['VwiC3b', 'yXK7lf'])[:3]:
                snippets.append(result.get_text().lower())

            combined_text = ' '.join(snippets)

            # Detect all industry keywords present
            detected_industries = [
                keyword for keyword in self.industry_keywords
                if keyword in combined_text
            ]

            return {
                'enriched': True,
                'industries': detected_industries,
                'context': combined_text[:500] if combined_text else ''
            }
        except Exception as e:
            return {
                'enriched': False,
                'industries': [],
                'context': '',
                'error': str(e)
            }

    def load_and_index_csv(self, csv_path: str, enrich_top_n: int = 100):
        """
        Load CSV and create enhanced FAISS index

        Args:
            csv_path: Path to the sponsor CSV file
            enrich_top_n: Number of top companies to enrich with web data (default 100)
        """
        try:
            self.sponsors_df = pd.read_csv(csv_path, encoding='utf-8')
            self.sponsors_df.columns = self.sponsors_df.columns.str.strip()

            # Filter for Skilled Worker route
            if 'Route' in self.sponsors_df.columns:
                self.sponsors_df = self.sponsors_df[
                    self.sponsors_df['Route'].str.contains('Skilled Worker', case=False, na=False)
                ]

            print(f"✅ Loaded {len(self.sponsors_df)} Skilled Worker sponsors")
            print("🔄 Creating Enhanced FAISS index with semantic embeddings...")

            # Prepare documents and metadata with ENHANCED descriptions
            documents = []
            metadatas = []

            for idx, row in self.sponsors_df.iterrows():
                company_name = str(row['Organisation Name'])

                # Create rich document text for better semantic search
                doc_parts = [company_name]

                # Add location
                if pd.notna(row.get('Town/City')):
                    doc_parts.append(f"located in {row['Town/City']}")

                if pd.notna(row.get('County')):
                    doc_parts.append(f"{row['County']}")

                # Add rating info
                if pd.notna(row.get('Type & Rating')):
                    doc_parts.append(f"rating {row['Type & Rating']}")

                # Create base document
                doc_text = ' '.join(doc_parts)

                # Detect potential industry keywords by name patterns
                name_lower = company_name.lower()
                potential_industry_match = any(keyword in name_lower for keyword in self.industry_keywords)

                # Enrich top companies or suspected industry-specific companies
                enrichment_data = {'enriched': False, 'industries': [], 'context': ''}

                if idx < enrich_top_n or potential_industry_match:
                    if idx % 10 == 0:  # Print progress every 10 companies
                        print(f"   Enriching company {idx+1}/{min(len(self.sponsors_df), enrich_top_n)}...")
                    enrichment_data = self._enrich_company_data(company_name)

                # Add enriched context to document for better embedding
                if enrichment_data.get('context'):
                    doc_text += f" {enrichment_data['context'][:200]}"

                documents.append(doc_text)

                metadatas.append({
                    "name": company_name,
                    "city": str(row.get('Town/City', '')),
                    "county": str(row.get('County', '')),
                    "rating": str(row.get('Type & Rating', '')),
                    "route": str(row.get('Route', '')),
                    "industries": enrichment_data.get('industries', []),
                    "enriched": enrichment_data.get('enriched', False)
                })

            print("   Encoding documents with semantic embeddings...")
            embeddings = self.embedding_model.encode(
                documents,
                show_progress=True,
                batch_size=32
            )

            print("   Building FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(self.dimension)
            self.faiss_index.add(embeddings.astype('float32'))

            self.id_to_metadata = {i: meta for i, meta in enumerate(metadatas)}

            # Cache everything
            self._save_to_cache(documents, embeddings, metadatas)

            print(f"✅ Indexed {self.faiss_index.ntotal} sponsors with Enhanced FAISS")
            enriched_count = sum(1 for m in metadatas if m.get('enriched'))
            print(f"   • Companies enriched with industry data: {enriched_count}")

        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            raise

    def _save_to_cache(self, documents: List[str], embeddings, metadatas: List[Dict]):
        """Save database to cache files"""
        print("💾 Caching Enhanced FAISS index and metadata...")

        faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE))

        with open(METADATA_CACHE_FILE, 'wb') as f:
            pickle.dump(self.id_to_metadata, f)

        cache_data = {
            'documents': documents,
            'embeddings': embeddings.tolist(),
            'metadatas': metadatas,
        }

        with open(EMBEDDINGS_CACHE_FILE, 'wb') as f:
            pickle.dump(cache_data, f)

        with open(DATA_CACHE_FILE, 'wb') as f:
            pickle.dump(self.sponsors_df.to_csv().encode('utf-8'), f)

    def search_with_semantic_filtering(self, query: str, max_results: int = 15,
                                      filter_a_rated_only: bool = False,
                                      use_intelligent_filtering: bool = None) -> Dict[str, Any]:
        """
        Enhanced semantic search with intelligent filtering

        Args:
            query: Search query string
            max_results: Maximum number of results
            filter_a_rated_only: Filter for A-rated sponsors only
            use_intelligent_filtering: Use LLM to filter results (None = use config default)

        Returns:
            Dictionary with search results
        """
        try:
            # Validate inputs
            max_results = self._validate_max_results(max_results)
            filter_a_rated_only = self._validate_boolean(filter_a_rated_only)

            # Use config default if not specified
            if use_intelligent_filtering is None:
                from config import USE_LLM_FILTERING
                use_intelligent_filtering = USE_LLM_FILTERING

            if self.faiss_index is None or self.faiss_index.ntotal == 0:
                return {
                    "success": False,
                    "error": "FAISS index not initialized. Please load the CSV file first."
                }

            if not query or not isinstance(query, str):
                return {
                    "success": False,
                    "error": "Invalid query. Please provide a search string."
                }

            # Check if query mentions any industry keywords
            query_lower = query.lower()
            query_industries = [
                keyword for keyword in self.industry_keywords
                if keyword in query_lower
            ]
            is_industry_specific_query = len(query_industries) > 0

            # Encode query
            query_embedding = self.embedding_model.encode_query(query)

            # Search with larger k for filtering
            k = max_results * 5 if (filter_a_rated_only or is_industry_specific_query) else max_results * 2
            k = min(k, self.faiss_index.ntotal)

            distances, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1),
                k
            )

            # Process results with intelligent filtering
            sponsors = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                metadata = self.id_to_metadata.get(int(idx), {})
                company_industries = metadata.get('industries', [])

                sponsor = {
                    'name': metadata.get('name', ''),
                    'city': metadata.get('city', ''),
                    'rating': metadata.get('rating', ''),
                    'industries': company_industries,
                    'relevance_score': float(distance)
                }

                # Apply filters
                if filter_a_rated_only and 'A rating' not in sponsor['rating']:
                    continue

                # If industry-specific query, boost companies matching those industries
                if is_industry_specific_query and company_industries:
                    # Check if company has any of the queried industries
                    industry_match = any(
                        q_ind in company_industries for q_ind in query_industries
                    )
                    if industry_match:
                        # Boost relevance for industry matches
                        sponsor['relevance_score'] *= 1.3
                    else:
                        # Slight penalty for non-matching industries
                        sponsor['relevance_score'] *= 0.8

                sponsors.append(sponsor)

            # Sort by relevance
            sponsors.sort(key=lambda x: x['relevance_score'], reverse=True)

            # Use LLM for intelligent filtering if enabled
            if use_intelligent_filtering and self.gemini_model and len(sponsors) > max_results:
                sponsors = self._llm_filter_results(query, sponsors[:max_results * 2], max_results)
            else:
                sponsors = sponsors[:max_results]

            return {
                "success": True,
                "count": len(sponsors),
                "sponsors": sponsors,
                "query_used": query,
                "detected_industries": query_industries
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def _llm_filter_results(self, query: str, candidates: List[Dict], max_results: int) -> List[Dict]:
        """
        Use LLM to intelligently filter and rank results

        Args:
            query: User's search query
            candidates: List of candidate sponsors
            max_results: Maximum results to return

        Returns:
            Filtered and ranked list of sponsors
        """
        if not self.gemini_model:
            return candidates[:max_results]

        try:
            # Create prompt for LLM
            companies_list = "\n".join([
                f"{i+1}. {s['name']} (City: {s['city']}, Industries: {', '.join(s.get('industries', [])[:3]) if s.get('industries') else 'Unknown'})"
                for i, s in enumerate(candidates[:30])  # Limit to top 30 for token efficiency
            ])

            prompt = f"""Given the user query: "{query}"

Analyze these companies and select the {max_results} MOST RELEVANT ones:

{companies_list}

Consider:
1. Direct name matches
2. Industry relevance (match query intent to company industries)
3. Location matches
4. Sector alignment

Examples:
- "technology companies" → prioritize tech, fintech, software
- "healthcare" → prioritize medical, pharmaceutical, biotech
- "finance" → prioritize banking, investment, insurance
- "consulting" → prioritize advisory, professional services

Return ONLY a JSON array of company numbers (1-indexed) in order of relevance:
Example: [3, 1, 15, 7, 22]

Your response:"""

            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                ),
                safety_settings={
                    'HARASSMENT': 'block_none',
                    'HATE_SPEECH': 'block_none',
                    'SEXUALLY_EXPLICIT': 'block_none',
                    'DANGEROUS_CONTENT': 'block_none'
                }
            )

            # Check if response was blocked
            if not response.candidates or not hasattr(response.candidates[0], 'content'):
                print(f"   ⚠️ LLM response blocked (safety filters), using semantic ranking")
                return candidates[:max_results]

            # Parse LLM response
            import json
            import re

            try:
                result_text = response.text.strip()
            except ValueError as e:
                # Response doesn't have text (blocked or empty)
                print(f"   ⚠️ LLM response empty, using semantic ranking")
                return candidates[:max_results]

            # Extract JSON array
            json_match = re.search(r'\[([\d,\s]+)\]', result_text)

            if json_match:
                selected_indices = json.loads(json_match.group(0))
                # Convert to 0-indexed and filter
                filtered = []
                for idx in selected_indices[:max_results]:
                    if 1 <= idx <= len(candidates):
                        filtered.append(candidates[idx - 1])

                if filtered:
                    print(f"   ✓ LLM filtered {len(candidates)} → {len(filtered)} results")
                    return filtered

        except Exception as e:
            print(f"   ⚠️ LLM filtering error: {str(e)[:100]}")

        # Fallback to original ranking
        return candidates[:max_results]

    def search(self, query: str, max_results: int = 15,
               filter_a_rated_only: bool = False) -> Dict[str, Any]:
        """Wrapper for backward compatibility - uses config default for LLM filtering"""
        return self.search_with_semantic_filtering(
            query, max_results, filter_a_rated_only, use_intelligent_filtering=None
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.sponsors_df is None:
            return {"success": False, "error": "Database not loaded"}

        enriched_count = sum(1 for m in self.id_to_metadata.values() if m.get('enriched', False))

        # Count companies with detected industries
        companies_with_industries = sum(
            1 for m in self.id_to_metadata.values()
            if m.get('industries') and len(m.get('industries', [])) > 0
        )

        return {
            "success": True,
            "total_sponsors": len(self.sponsors_df),
            "a_rated": len(self.sponsors_df[
                self.sponsors_df['Type & Rating'].str.contains('A rating', na=False)
            ]) if 'Type & Rating' in self.sponsors_df.columns else 0,
            "unique_cities": self.sponsors_df['Town/City'].nunique() if 'Town/City' in self.sponsors_df.columns else 0,
            "enriched_companies": enriched_count,
            "companies_with_industries": companies_with_industries
        }

    @staticmethod
    def _validate_max_results(max_results) -> int:
        """Validate and convert max_results parameter"""
        try:
            max_results = int(max_results) if max_results is not None else 15
        except (ValueError, TypeError):
            max_results = 15
        return max(1, min(max_results, 50))

    @staticmethod
    def _validate_boolean(value) -> bool:
        """Validate and convert boolean parameter"""
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes')
        return bool(value) if value is not None else False
